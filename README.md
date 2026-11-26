#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

struct ReuseUBBuffersPass
    : public PassWrapper<ReuseUBBuffersPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReuseUBBuffersPass)

  StringRef getArgument() const final { return "hivm-reuse-ub-buffers"; }
  StringRef getDescription() const final {
    return "Eliminates intermediate 2D UB buffer between adjacent pipeline stages.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool changed = false;

    // Обходим весь ModuleOp (работает для любого типа функций: func.func, hivm.func и т.д.)
    module.walk([&](scf::ForOp mainLoop) {
      SmallVector<scf::IfOp, 8> ifOps;
      for (Operation &op : mainLoop.getBody()->without_terminator()) {
        if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
          ifOps.push_back(ifOp);
        }
      }

      if (ifOps.size() < 2)
        return;

      for (size_t i = 0; i + 1 < ifOps.size(); ++i) {
        scf::IfOp producerIf = ifOps[i];
        scf::IfOp consumerIf = ifOps[i + 1];

        Block &prodBlock = producerIf.getThenRegion().front();
        Block &consBlock = consumerIf.getThenRegion().front();

        // 1. Ищем операцию записи 2D f32 тензора в memref внутри producerIf
        Operation *copyToTempOp = nullptr;
        Value tempMemrefCast = nullptr;
        Value scaledTensor = nullptr;

        for (Operation &op : prodBlock) {
          StringRef opName = op.getName().getStringRef();
          if (opName.contains("copy") && op.getNumOperands() >= 2) {
            Value src = op.getOperand(0);
            Value dst = op.getOperand(1);
            if (auto tensorType = dyn_cast<RankedTensorType>(src.getType())) {
              if (tensorType.getRank() == 2 &&
                  tensorType.getElementType().isF32() &&
                  isa<MemRefType>(dst.getType())) {
                copyToTempOp = &op;
                scaledTensor = src;
                tempMemrefCast = dst;
                break;
              }
            }
          }
        }

        if (!copyToTempOp)
          continue;

        // 2. Ищем arith.mulf, породивший scaledTensor
        auto mulOp = scaledTensor.getDefiningOp<arith::MulFOp>();
        if (!mulOp) {
          llvm::errs() << "[ReuseUBBuffersPass] Skip: copy src is not arith.mulf, it is "
                       << scaledTensor.getDefiningOp()->getName() << "\n";
          continue;
        }

        bufferization::ToTensorOp toTensorSrc = nullptr;
        Value scaleTensor = nullptr;

        if (auto ttLhs = mulOp.getLhs().getDefiningOp<bufferization::ToTensorOp>()) {
          toTensorSrc = ttLhs;
          scaleTensor = mulOp.getRhs();
        } else if (auto ttRhs = mulOp.getRhs().getDefiningOp<bufferization::ToTensorOp>()) {
          toTensorSrc = ttRhs;
          scaleTensor = mulOp.getLhs();
        }

        if (!toTensorSrc) {
          llvm::errs() << "[ReuseUBBuffersPass] Skip: mul operand is not bufferization.to_tensor\n";
          continue;
        }

        auto castSrc = toTensorSrc.getBuffer().getDefiningOp<memref::MemorySpaceCastOp>();
        if (!castSrc) {
          llvm::errs() << "[ReuseUBBuffersPass] Skip: to_tensor buffer is not memory_space_cast\n";
          continue;
        }
        Value origAlloc = castSrc.getSource();

        // 3. Ищем чтение из tempMemrefCast в consumerIf
        bufferization::ToTensorOp consumerToTensor = nullptr;
        for (Operation &op : consBlock) {
          if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(&op)) {
            if (toTensor.getBuffer() == tempMemrefCast) {
              consumerToTensor = toTensor;
              break;
            }
          }
        }

        if (!consumerToTensor) {
          llvm::errs() << "[ReuseUBBuffersPass] Skip: consumer stage does not read temp buffer\n";
          continue;
        }

        // 4. Ищем sync_block_set с flag = 4 в producerIf
        SmallVector<Operation *, 4> syncSetOps;
        for (Operation &op : prodBlock) {
          if (op.getName().getStringRef().contains("sync_block_set")) {
            if (auto flagAttr = op.getAttrOfType<IntegerAttr>("flag")) {
              if (flagAttr.getInt() == 4) {
                syncSetOps.push_back(&op);
              }
            }
          }
        }

        // 5. Заменяем чтение из temp-буфера на чтение из origAlloc + рекомпутацию
        OpBuilder builder(consumerToTensor);
        Location loc = consumerToTensor.getLoc();

        auto newCast = builder.create<memref::MemorySpaceCastOp>(
            loc, consumerToTensor.getBuffer().getType(), origAlloc);
        auto newToTensor = builder.create<bufferization::ToTensorOp>(
            loc, consumerToTensor.getType(), newCast, /*restrict=*/true, /*writable=*/true);

        auto recomputedMul = builder.create<arith::MulFOp>(loc, newToTensor, scaleTensor);

        consumerToTensor.replaceAllUsesWith(recomputedMul.getResult());
        consumerToTensor.erase();

        for (Operation *syncOp : syncSetOps) {
          syncOp->moveAfter(recomputedMul);
        }

        copyToTempOp->erase();

        if (auto tempCastOp = tempMemrefCast.getDefiningOp<memref::MemorySpaceCastOp>()) {
          Value tempAlloc = tempCastOp.getSource();
          if (tempCastOp->use_empty())
            tempCastOp->erase();
          if (Operation *allocOp = tempAlloc.getDefiningOp()) {
            if (allocOp->use_empty())
              allocOp->erase();
          }
        }

        llvm::errs() << "[ReuseUBBuffersPass] SUCCESS: Eliminated 128x128 UB buffer!\n";
        changed = true;
      }
    });

    if (!changed) {
      llvm::errs() << "[ReuseUBBuffersPass] Finished on ModuleOp with NO changes.\n";
    }
  }
};

} // namespace

std::unique_ptr<Pass> createReuseUBBuffersPass() {
  return std::make_unique<ReuseUBBuffersPass>();
}
