#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include <memory>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOptPass.h"
#include "ascend/include/DynamicCVPipeline/MainLoopUnroll.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlockPass.h"
#include "ascend/include/DynamicCVPipeline/PreCheckAvailable.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflowPass.h"
#include "ascend/include/DynamicCVPipeline/StandardizeOp.h"

using namespace mlir;

namespace {

struct HoistReductionAcrossStagesPass
    : public PassWrapper<HoistReductionAcrossStagesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HoistReductionAcrossStagesPass)

  StringRef getArgument() const final { return "hivm-hoist-stage-reduction"; }
  StringRef getDescription() const final {
    return "Hoists linalg.reduce to producer stage to eliminate 2D UB buffers";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(module.getContext());

    SmallVector<memref::AllocOp> candidateAllocs;
    module.walk([&](memref::AllocOp allocOp) {
      auto memrefType = allocOp.getType();
      if (memrefType.getRank() == 2 && memrefType.getMemorySpace()) {
        candidateAllocs.push_back(allocOp);
      }
    });

    for (memref::AllocOp allocOp : candidateAllocs) {
      tryOptimizeBuffer(allocOp, rewriter);
    }
  }

private:
  void tryOptimizeBuffer(memref::AllocOp allocOp, IRRewriter &rewriter) {
    if (!allocOp->hasOneUse())
      return;
    auto castOp = dyn_cast<memref::MemorySpaceCastOp>(*allocOp->getUsers().begin());
    if (!castOp)
      return;

    Operation *writeOp = nullptr;
    bufferization::ToTensorOp readOp = nullptr;

    for (Operation *user : castOp->getUsers()) {
      if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(user)) {
        if (readOp) return;
        readOp = toTensor;
      } else if (user->getName().getStringRef() == "hivm.hir.copy") {
        if (user->getOperand(1) == castOp.getResult()) {
          if (writeOp) return;
          writeOp = user;
        }
      } else {
        return;
      }
    }

    if (!writeOp || !readOp || !readOp->hasOneUse())
      return;

    auto reduceOp = dyn_cast<linalg::ReduceOp>(*readOp->getUsers().begin());
    if (!reduceOp)
      return;

    if (reduceOp.getInputs().size() != 1 || reduceOp.getInputs()[0] != readOp.getResult())
      return;

    auto fillOp = reduceOp.getInits()[0].getDefiningOp<linalg::FillOp>();
    if (!fillOp)
      return;
    auto emptyOp = fillOp.output().getDefiningOp<tensor::EmptyOp>();
    if (!emptyOp)
      return;

    if (writeOp->getBlock() == reduceOp->getBlock())
      return;

    rewriter.setInsertionPoint(allocOp);
    auto reducedTensorType = cast<RankedTensorType>(reduceOp.getResult(0).getType());
    auto newMemrefType = MemRefType::get(
        reducedTensorType.getShape(),
        reducedTensorType.getElementType(),
        MemRefLayoutAttrInterface{},
        allocOp.getType().getMemorySpace());

    auto newAlloc = rewriter.create<memref::AllocOp>(allocOp.getLoc(), newMemrefType);
    auto genericNewMemrefType = MemRefType::get(
        reducedTensorType.getShape(), reducedTensorType.getElementType());
    auto newCast = rewriter.create<memref::MemorySpaceCastOp>(
        castOp.getLoc(), genericNewMemrefType, newAlloc);

    rewriter.setInsertionPoint(writeOp);
    Value producer2DTensor = writeOp->getOperand(0);

    IRMapping mapping;
    Operation *clonedEmpty = rewriter.clone(*emptyOp, mapping);
    mapping.map(emptyOp.getResult(), clonedEmpty->getResult(0));

    Operation *clonedFill = rewriter.clone(*fillOp, mapping);
    mapping.map(fillOp.getResult(0), clonedFill->getResult(0));
    mapping.map(readOp.getResult(), producer2DTensor);

    Operation *clonedReduce = rewriter.clone(*reduceOp, mapping);
    Value reduced1DTensor = clonedReduce->getResult(0);

    rewriter.modifyOpInPlace(writeOp, [&]() {
      writeOp->setOperand(0, reduced1DTensor);
      writeOp->setOperand(1, newCast.getResult());
    });

    rewriter.setInsertionPoint(readOp);
    auto newToTensor = rewriter.create<bufferization::ToTensorOp>(
        readOp.getLoc(), reducedTensorType, newCast.getResult(),
        readOp.getRestrict(), readOp.getWritable());

    rewriter.replaceOp(reduceOp, newToTensor.getResult());

    rewriter.eraseOp(readOp);
    if (fillOp->use_empty()) rewriter.eraseOp(fillOp);
    if (emptyOp->use_empty()) rewriter.eraseOp(emptyOp);
    rewriter.eraseOp(castOp);
    rewriter.eraseOp(allocOp);
  }
};

} // namespace

std::unique_ptr<Pass> createHoistReductionAcrossStagesPass() {
  return std::make_unique<HoistReductionAcrossStagesPass>();
}
