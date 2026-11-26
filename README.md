#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

static bool isUBMemRef(Value val) {
  auto memrefTy = dyn_cast<MemRefType>(val.getType());
  if (!memrefTy || !memrefTy.getMemorySpace())
    return false;
  std::string spaceStr;
  llvm::raw_string_ostream os(spaceStr);
  memrefTy.getMemorySpace().print(os);
  return os.str().find("ub") != std::string::npos;
}

static bool isTemporary2DUBAlloc(memref::AllocOp alloc) {
  if (!isUBMemRef(alloc.getResult()))
    return false;
  auto ty = alloc.getType();
  ArrayRef<int64_t> shape = ty.getShape();
  if (ty.getRank() != 2 || shape.front() != 128 || shape.back() != 128)
    return false;
  if (!ty.getElementType().isF32())
    return false;

  for (Operation *user : alloc->getUsers()) {
    if (user->getName().getStringRef().contains("annotation.mark"))
      return false;
  }
  return true;
}

static memref::AllocOp getUnderlyingAlloc(Value memrefVal) {
  while (auto castOp = memrefVal.getDefiningOp<memref::MemorySpaceCastOp>()) {
    memrefVal = castOp.getSource();
  }
  return memrefVal.getDefiningOp<memref::AllocOp>();
}

static bool isSyncFlag4(Operation *op) {
  if (op->getName().getStringRef() != "hivm.hir.sync_block_set")
    return false;
  std::string opStr;
  llvm::raw_string_ostream os(opStr);
  op->print(os);
  return os.str().find("flag = 4") != std::string::npos;
}


struct OptimizeUBAllocationsPass
    : public PassWrapper<OptimizeUBAllocationsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizeUBAllocationsPass)

  StringRef getArgument() const final { return "optimize-hivm-ub-allocs"; }
  StringRef getDescription() const final {
    return "Сокращает все промежуточные 2D ping-pong буферы в UB";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](scf::ForOp forOp) {
      if (!forOp->hasAttr("ssbuffer.main_loop_probe"))
        return;

      optimize2DPingPongReductions(forOp);
      eliminateS2PingPongBuffers(forOp);
      eraseDead2DAllocs(forOp);
    });
  }

private:
  void optimize2DPingPongReductions(scf::ForOp forOp) {
    OpBuilder builder(forOp);

    SmallVector<linalg::ReduceOp> targetReduces;
    forOp.walk([&](linalg::ReduceOp redOp) {
      ArrayRef<int64_t> dims = redOp.getDimensions();
      if (dims.size() != 1 || dims.front() != 1)
        return;

      Value input = redOp.getInputs().front();
      auto tensorTy = dyn_cast<RankedTensorType>(input.getType());
      if (!tensorTy || tensorTy.getRank() != 2 ||
          tensorTy.getShape().front() != 128 ||
          tensorTy.getShape().back() != 128)
        return;

      if (auto ifOp = input.getDefiningOp<scf::IfOp>()) {
        unsigned resIdx = cast<OpResult>(input).getResultNumber();
        auto thenYield = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
        auto elseYield = cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());

        auto thenToTensor = thenYield.getOperand(resIdx)
                                .getDefiningOp<bufferization::ToTensorOp>();
        auto elseToTensor = elseYield.getOperand(resIdx)
                                .getDefiningOp<bufferization::ToTensorOp>();

        if (!thenToTensor || !elseToTensor)
          return;

        auto alloc0 = getUnderlyingAlloc(thenToTensor.getBuffer());
        auto alloc1 = getUnderlyingAlloc(elseToTensor.getBuffer());
        if (alloc0 && alloc1 && isTemporary2DUBAlloc(alloc0) &&
            isTemporary2DUBAlloc(alloc1)) {
          targetReduces.push_back(redOp);
        }
      }
    });

    for (linalg::ReduceOp reduceOp : targetReduces) {
      Value input = reduceOp.getInputs().front();
      auto ifOp = input.getDefiningOp<scf::IfOp>();
      unsigned resIdx = cast<OpResult>(input).getResultNumber();

      auto thenYield = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
      auto elseYield = cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());

      auto thenToTensor = thenYield.getOperand(resIdx)
                              .getDefiningOp<bufferization::ToTensorOp>();
      auto elseToTensor = elseYield.getOperand(resIdx)
                              .getDefiningOp<bufferization::ToTensorOp>();

      SmallVector<bufferization::ToTensorOp, 2> branchReaders = {thenToTensor,
                                                                 elseToTensor};
      SmallVector<Value, 2> new1DCasts;

      for (bufferization::ToTensorOp reader : branchReaders) {
        memref::AllocOp oldAlloc = getUnderlyingAlloc(reader.getBuffer());
        auto ubSpace = oldAlloc.getType().getMemorySpace();
        auto new1DMemRefType = MemRefType::get(
            {128}, builder.getF32Type(), MemRefLayoutAttrInterface(), ubSpace);
        auto default1DType = MemRefType::get({128}, builder.getF32Type());

        builder.setInsertionPoint(oldAlloc);
        auto newAlloc1D =
            builder.create<memref::AllocOp>(oldAlloc.getLoc(), new1DMemRefType);
        auto newCast1D = builder.create<memref::MemorySpaceCastOp>(
            oldAlloc.getLoc(), default1DType, newAlloc1D);
        new1DCasts.push_back(newCast1D);

        SmallVector<Operation *> copyWriters;
        for (Operation *user : oldAlloc->getUsers()) {
          if (auto cast = dyn_cast<memref::MemorySpaceCastOp>(user)) {
            for (Operation *castUser : cast->getUsers()) {
              if (castUser->getName().getStringRef() == "hivm.hir.copy" &&
                  castUser->getNumOperands() >= 2 &&
                  castUser->getOperand(1) == cast.getResult()) {
                copyWriters.push_back(castUser);
              }
            }
          }
        }

        for (Operation *copyOp : copyWriters) {
          builder.setInsertionPoint(copyOp);
          Value expTensor2D = copyOp->getOperand(0);

          auto dstTensor = builder.create<bufferization::ToTensorOp>(
              copyOp->getLoc(),
              RankedTensorType::get({128}, builder.getF32Type()), newCast1D,
              /*restrict=*/true, /*writable=*/true);

          Value zeroConst = builder.create<arith::ConstantOp>(
              copyOp->getLoc(), builder.getF32FloatAttr(0.0f));
          auto filled1D = builder.create<linalg::FillOp>(
              copyOp->getLoc(), zeroConst, dstTensor.getResult());

          IRMapping mapper;
          mapper.map(reduceOp.getInputs().front(), expTensor2D);
          mapper.map(reduceOp.getInits().front(), filled1D.getResult(0));
          auto *clonedReduce = builder.clone(*reduceOp, mapper);

          copyOp->setOperand(0, clonedReduce->getResult(0));
          copyOp->setOperand(1, newCast1D);
        }
      }

      builder.setInsertionPoint(ifOp);
      auto newIfOp = builder.create<scf::IfOp>(
          ifOp.getLoc(),
          TypeRange{RankedTensorType::get({128}, builder.getF32Type())},
          ifOp.getCondition(), /*withElseRegion=*/true);

      builder.setInsertionPointToStart(newIfOp.thenBlock());
      auto then1D = builder.create<bufferization::ToTensorOp>(
          thenToTensor.getLoc(),
          RankedTensorType::get({128}, builder.getF32Type()),
          new1DCasts.front(), /*restrict=*/true, /*writable=*/true);
      builder.create<scf::YieldOp>(thenYield.getLoc(), then1D.getResult());

      builder.setInsertionPointToStart(newIfOp.elseBlock());
      auto else1D = builder.create<bufferization::ToTensorOp>(
          elseToTensor.getLoc(),
          RankedTensorType::get({128}, builder.getF32Type()),
          new1DCasts.back(), /*restrict=*/true, /*writable=*/true);
      builder.create<scf::YieldOp>(elseYield.getLoc(), else1D.getResult());

      reduceOp.getResult(0).replaceAllUsesWith(newIfOp.getResult(0));
      reduceOp.erase();
      ifOp.erase();
    }
  }

    void eliminateS2PingPongBuffers(scf::ForOp forOp) {
    scf::IfOp stage15If = nullptr;
    scf::IfOp stage16If = nullptr;

    forOp.walk([&](scf::IfOp ifOp) {
      if (auto attr = ifOp->getAttrOfType<IntegerAttr>("ssbuffer.if")) {
        if (attr.getInt() == 15)
          stage15If = ifOp;
        else if (attr.getInt() == 16)
          stage16If = ifOp;
      }
    });

    if (!stage15If || !stage16If)
      return;

    scf::IfOp s2ReaderIf = nullptr;
    memref::AllocOp s2Alloc0 = nullptr;
    memref::AllocOp s2Alloc1 = nullptr;

    stage16If.thenBlock()->walk([&](scf::IfOp innerIf) {
      if (innerIf.getNumResults() != 1)
        return;
      auto resTy = dyn_cast<RankedTensorType>(innerIf.getResult(0).getType());
      if (!resTy || resTy.getRank() != 2 || resTy.getShape().front() != 128 ||
          resTy.getShape().back() != 128)
        return;

      auto thenYield = cast<scf::YieldOp>(innerIf.thenBlock()->getTerminator());
      auto elseYield = cast<scf::YieldOp>(innerIf.elseBlock()->getTerminator());
      auto t0 =
          thenYield.getOperand(0).getDefiningOp<bufferization::ToTensorOp>();
      auto t1 =
          elseYield.getOperand(0).getDefiningOp<bufferization::ToTensorOp>();
      if (!t0 || !t1)
        return;

      auto a0 = getUnderlyingAlloc(t0.getBuffer());
      auto a1 = getUnderlyingAlloc(t1.getBuffer());
      if (a0 && a1 && isTemporary2DUBAlloc(a0) && isTemporary2DUBAlloc(a1)) {
        s2ReaderIf = innerIf;
        s2Alloc0 = a0;
        s2Alloc1 = a1;
      }
    });

    if (!s2ReaderIf)
      return;

    Operation *setFlag4 = nullptr;
    arith::MulFOp s2MulOp = nullptr;
    SmallVector<scf::IfOp> writerIfsToErase;

    for (Operation &op : *stage15If.thenBlock()) {
      if (isSyncFlag4(&op)) {
        setFlag4 = &op;
      } else if (auto innerIf = dyn_cast<scf::IfOp>(&op)) {
        bool writesS2 = false;
        innerIf.walk([&](Operation *subOp) {
          if (subOp->getName().getStringRef() == "hivm.hir.copy" &&
              subOp->getNumOperands() >= 2) {
            auto dstAlloc = getUnderlyingAlloc(subOp->getOperand(1));
            if (dstAlloc == s2Alloc0 || dstAlloc == s2Alloc1) {
              writesS2 = true;
              if (auto mul =
                      subOp->getOperand(0).getDefiningOp<arith::MulFOp>()) {
                s2MulOp = mul;
              }
            }
          }
        });
        if (writesS2)
          writerIfsToErase.push_back(innerIf);
      }
    }

    if (!setFlag4 || !s2MulOp)
      return;

    auto origToTensor =
        s2MulOp.getLhs().getDefiningOp<bufferization::ToTensorOp>();
    if (!origToTensor)
      return;
    memref::AllocOp tightlyCoupledAlloc18 =
        getUnderlyingAlloc(origToTensor.getBuffer());
    if (!tightlyCoupledAlloc18)
      return;

    OpBuilder builder(s2ReaderIf);

    auto cast18 = builder.create<memref::MemorySpaceCastOp>(
        s2ReaderIf.getLoc(), MemRefType::get({128, 128}, builder.getF32Type()),
        tightlyCoupledAlloc18);
    auto read18 = builder.create<bufferization::ToTensorOp>(
        s2ReaderIf.getLoc(),
        RankedTensorType::get({128, 128}, builder.getF32Type()), cast18,
        /*restrict=*/true, /*writable=*/true);

    auto recomputedS2 = builder.create<arith::MulFOp>(
        s2ReaderIf.getLoc(), read18.getResult(), s2MulOp.getRhs());

    Operation *lastUserIn16 = recomputedS2;
    for (Operation *u : s2ReaderIf.getResult(0).getUsers()) {
      if (u->getBlock() == stage16If.thenBlock() &&
          lastUserIn16->isBeforeInBlock(u)) {
        lastUserIn16 = u;
      }
    }
    builder.setInsertionPointAfter(lastUserIn16);
    builder.clone(*setFlag4);

    s2ReaderIf.getResult(0).replaceAllUsesWith(recomputedS2.getResult());
    s2ReaderIf.erase();

    for (scf::IfOp wIf : writerIfsToErase)
      wIf.erase();

    setFlag4->erase();
  }

  void eraseDead2DAllocs(scf::ForOp forOp) {
    Block *parentBlock = forOp->getBlock();
    SmallVector<Operation *> castsToErase;
    SmallVector<Operation *> allocsToErase;

    for (Operation &op : *parentBlock) {
      if (auto alloc = dyn_cast<memref::AllocOp>(&op)) {
        if (!isTemporary2DUBAlloc(alloc))
          continue;
        bool allDead = true;
        for (Operation *u : alloc->getUsers()) {
          if (!isa<memref::MemorySpaceCastOp>(u) || !u->use_empty()) {
            allDead = false;
            break;
          }
        }
        if (allDead) {
          for (Operation *u : alloc->getUsers())
            castsToErase.push_back(u);
          allocsToErase.push_back(alloc);
        }
      }
    }
    for (Operation *op : castsToErase)
      op->erase();
    for (Operation *op : allocsToErase)
      op->erase();

    llvm::errs() << "[OptimizeUBAllocationsPass] Erased total 2D UB allocs: "
                 << allocsToErase.size() << "\n";
  }
};

} // namespace

std::unique_ptr<Pass> createOptimizeUBAllocationsPass() {
  return std::make_unique<OptimizeUBAllocationsPass>();
}
