#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

/// Проверяет, используется ли тензор только как временный буфер (outs)
/// в операциях, которые полностью перезаписывают его перед чтением,
/// либо вообще не имеет читателей (мертвый код).
static bool isDeadTensor(Value val) {
  for (OpOperand &use : val.getUses()) {
    Operation *owner = use.getOwner();
    // Если результат передается в linalg.fill, проверяем пользователей самого fill
    if (auto fillOp = dyn_cast<linalg::FillOp>(owner)) {
      if (use.getOperandNumber() == 1) { // outs operand
        if (!isDeadTensor(fillOp.getResult(0)))
          return false;
        continue;
      }
    }
    // Любое другое использование означает, что тензор живой
    return false;
  }
  return true;
}

/// Удаляет мертвые linalg.fill и tensor.empty снизу вверх
static void eraseDeadEmptyChain(tensor::EmptyOp emptyOp) {
  SmallVector<Operation *> toErase;
  for (OpOperand &use : emptyOp->getUses()) {
    if (auto fillOp = dyn_cast<linalg::FillOp>(use.getOwner())) {
      toErase.push_back(fillOp);
    }
  }
  for (Operation *op : toErase) {
    op->erase();
  }
  emptyOp->erase();
}

struct EliminateRedundantEmptyTensorsPass
    : public PassWrapper<EliminateRedundantEmptyTensorsPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      EliminateRedundantEmptyTensorsPass)

  StringRef getArgument() const final {
    return "eliminate-redundant-empty-tensors";
  }

  StringRef getDescription() const final {
    return "Удаляет мертвые tensor.empty и переиспользует дублирующиеся "
           "аллокации одного типа в пределах ModuleOp";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Шаг 1: Удаление мертвых tensor.empty и связанных с ними linalg.fill по всему модулю
    SmallVector<tensor::EmptyOp> deadEmpties;
    module.walk([&](tensor::EmptyOp emptyOp) {
      if (isDeadTensor(emptyOp.getResult())) {
        deadEmpties.push_back(emptyOp);
      }
    });

    for (tensor::EmptyOp dead : deadEmpties) {
      eraseDeadEmptyChain(dead);
    }

    // Шаг 2: Дедупликация tensor.empty в пределах каждого базового блока (Block)
    module.walk([&](Block *block) {
      DenseMap<Type, tensor::EmptyOp> canonicalEmpties;
      SmallVector<tensor::EmptyOp> duplicatesToErase;

      for (Operation &op : *block) {
        auto emptyOp = dyn_cast<tensor::EmptyOp>(&op);
        if (!emptyOp)
          continue;

        RankedTensorType tensorType = emptyOp.getType();
        if (!tensorType.hasStaticShape())
          continue;

        auto it = canonicalEmpties.find(tensorType);
        if (it == canonicalEmpties.end()) {
          canonicalEmpties[tensorType] = emptyOp;
        } else {
          tensor::EmptyOp masterEmpty = it->second;
          emptyOp.getResult().replaceAllUsesWith(masterEmpty.getResult());
          duplicatesToErase.push_back(emptyOp);
        }
      }

      for (tensor::EmptyOp dup : duplicatesToErase) {
        dup->erase();
      }
    });

    // Шаг 3: Переиспользование доминирующих tensor.empty между вложенными регионами
    // Обрабатываем каждую операцию с IsolatedFromAbove (например, func.func или hivm kernel)
    // отдельно, так как SSA-значения не пересекают границы IsolatedFromAbove.
    module.walk([&](Operation *isolatedOp) {
      if (!isolatedOp->hasTrait<OpTrait::IsIsolatedFromAbove>() ||
          isa<ModuleOp>(isolatedOp))
        return;

      DominanceInfo domInfo(isolatedOp);
      SmallVector<tensor::EmptyOp> regionEmpties;

      isolatedOp->walk([&](tensor::EmptyOp op) {
        if (op.getType().hasStaticShape())
          regionEmpties.push_back(op);
      });

      for (size_t i = 0; i < regionEmpties.size(); ++i) {
        tensor::EmptyOp opA = regionEmpties[i];
        if (!opA || !opA->getBlock())
          continue;

        for (size_t j = i + 1; j < regionEmpties.size(); ++j) {
          tensor::EmptyOp opB = regionEmpties[j];
          if (!opB || !opB->getBlock())
            continue;

          if (opA.getType() == opB.getType()) {
            if (domInfo.dominates(opA.getOperation(), opB.getOperation())) {
              opB.getResult().replaceAllUsesWith(opA.getResult());
              opB->erase();
              regionEmpties[j] = nullptr;
            } else if (domInfo.dominates(opB.getOperation(),
                                         opA.getOperation())) {
              opA.getResult().replaceAllUsesWith(opB.getResult());
              opA->erase();
              regionEmpties[i] = nullptr;
              break;
            }
          }
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> createEliminateRedundantEmptyTensorsPass() {
  return std::make_unique<EliminateRedundantEmptyTensorsPass>();
}

static PassRegistration<EliminateRedundantEmptyTensorsPass> regPass;
