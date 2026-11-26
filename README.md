#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

static bool isDefinedOutsideLoop(Value val, scf::ForOp loop) {
  Operation *defOp = val.getDefiningOp();
  if (!defOp) {
    Block *parentBlock = cast<BlockArgument>(val).getOwner();
    return !loop->isAncestor(parentBlock->getParentOp());
  }
  return !loop->isAncestor(defOp);
}

static bool isDependentMetadataOrCastOp(Operation *op) {
  if (isa<memref::MemorySpaceCastOp>(op))
    return true;
  if (op->getName().getStringRef() == "annotation.mark")
    return true;
  return false;
}

struct HoistStaticAllocsPass
    : public PassWrapper<HoistStaticAllocsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HoistStaticAllocsPass)

  StringRef getArgument() const final { return "hoist-static-allocs"; }
  StringRef getDescription() const final {
    return "Выносит memref.alloc и связанные метаданные за пределы циклов scf.for";
  }

  void runOnOperation() override {
    ModuleOp func = getOperation();
    bool changed = true;

    while (changed) {
      changed = false;

      func.walk([&](scf::ForOp loop) {
        SmallVector<memref::AllocOp, 16> allocsToHoist;

        for (Operation &op : loop.getBody()->getOperations()) {
          auto allocOp = dyn_cast<memref::AllocOp>(&op);
          if (!allocOp)
            continue;

          bool allOperandsInvariant = llvm::all_of(
              allocOp->getOperands(),
              [&](Value operand) { return isDefinedOutsideLoop(operand, loop); });

          if (!allOperandsInvariant)
            continue;

          bool hasDeallocInLoop = false;
          for (Operation *user : allocOp->getUsers()) {
            if (isa<memref::DeallocOp>(user) && loop->isAncestor(user)) {
              hasDeallocInLoop = true;
              break;
            }
          }

          if (!hasDeallocInLoop) {
            allocsToHoist.push_back(allocOp);
          }
        }

        if (allocsToHoist.empty())
          return;

        for (memref::AllocOp allocOp : allocsToHoist) {
          SmallVector<Operation *, 4> dependentOps;
          for (Operation *user : allocOp->getUsers()) {
            if (user->getBlock() == allocOp->getBlock() &&
                isDependentMetadataOrCastOp(user)) {
                bool operandsSafe = llvm::all_of(user->getOperands(), [&](Value v) {
                return v == allocOp.getResult() || isDefinedOutsideLoop(v, loop);
              });
              if (operandsSafe) {
                dependentOps.push_back(user);
              }
            }
          }

          llvm::sort(dependentOps, [](Operation *a, Operation *b) {
            return a->isBeforeInBlock(b);
          });

          allocOp->moveBefore(loop);

          for (Operation *depOp : dependentOps) {
            depOp->moveBefore(loop);
          }

          changed = true;
        }
      });
    }
  }
};

} // namespace

std::unique_ptr<Pass> createHoistStaticAllocsPass() {
  return std::make_unique<HoistStaticAllocsPass>();
}
