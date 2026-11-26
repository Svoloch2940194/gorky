#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

class SplitSyncBufferLifetimePass
    : public PassWrapper<SplitSyncBufferLifetimePass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitSyncBufferLifetimePass)

  StringRef getArgument() const final {
    return "hivm-split-sync-buffer-lifetime";
  }
  StringRef getDescription() const final {
    return "Localizes shared tensor.empty outs in transpose/broadcast and "
           "sinks broadcast in ssbuffer.if=19 to save 32-64KB UB.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // 1. Заменяем общий %60 (tensor.empty) в linalg.transpose на локальный tensor.empty()
    // перед каждым transpose — это освобождает 32 КБ (128x128xf16), удерживаемые через весь цикл!
    module.walk([&](linalg::TransposeOp transposeOp) {
      Value init = transposeOp.getInit();
      auto initType = dyn_cast<RankedTensorType>(init.getType());
      if (!initType)
        return;

      if (init.getParentBlock() != transposeOp->getBlock() || !init.hasOneUse()) {
        OpBuilder builder(transposeOp);
        auto localEmpty = builder.create<tensor::EmptyOp>(
            transposeOp.getLoc(), initType.getShape(), initType.getElementType());
        transposeOp.getInitMutable().assign(localEmpty.getResult());
      }
    });

    // 2. В ssbuffer.if = 19 опускаем каждый linalg.broadcast прямо перед arith.mulf
    module.walk([&](scf::IfOp ifOp) {
      auto ssAttr = ifOp->getAttrOfType<IntegerAttr>("ssbuffer.if");
      if (!ssAttr || ssAttr.getInt() != 19)
        return;

      Block &thenBlock = ifOp.getThenRegion().front();
      SmallVector<linalg::BroadcastOp> bcasts;
      for (Operation &op : thenBlock) {
        if (auto bcast = dyn_cast<linalg::BroadcastOp>(&op))
          bcasts.push_back(bcast);
      }

      for (linalg::BroadcastOp bcast : bcasts) {
        if (!bcast->hasOneUse())
          continue;
        Operation *user = *bcast->getUsers().begin();
        if (user->getBlock() == &thenBlock && bcast->isBeforeInBlock(user)) {
          bcast->moveBefore(user);
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> createSplitSyncBufferLifetimePass() {
  return std::make_unique<SplitSyncBufferLifetimePass>();
}
