// FuseBroadcastIntoArithPass.cpp
//
// Устраняет полные broadcast-тензоры, сливая
//   linalg.broadcast + arith.subf/mulf/addf
// в единый linalg.generic. Устраняет источник temp_buffer,
// из-за которого PlanMemory раздувает UB.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

// Шаблон по типу arith-операции (SubFOp, MulFOp, AddFOp).
template <typename ArithOp>
struct FuseBroadcastIntoArith
    : public OpRewritePattern<linalg::BroadcastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BroadcastOp bcast,
                                PatternRewriter &rewriter) const override {
    // Гарантия безопасности: broadcast имеет ровно одного пользователя.
    Value bcastResult = bcast->getResult(0);
    if (!bcastResult.hasOneUse()) return failure();

    Operation *user = *bcastResult.getUsers().begin();
    if (!isa<ArithOp>(user)) return failure();

    // Какой операнд user'а — broadcast?
    Value fullOperand;
    bool bcastIsRhs;
    if (user->getOperand(0) == bcastResult) {
      fullOperand = user->getOperand(1);
      bcastIsRhs = false;
    } else if (user->getOperand(1) == bcastResult) {
      fullOperand = user->getOperand(0);
      bcastIsRhs = true;
    } else {
      return failure();
    }

    // Проверяем типы: только статические rank-1 + rank-N тензоры.
    auto bcastType = dyn_cast<RankedTensorType>(bcastResult.getType());
    auto srcType   = dyn_cast<RankedTensorType>(bcast.getInput().getType());
    auto fullType  = dyn_cast<RankedTensorType>(fullOperand.getType());
    if (!bcastType || !srcType || !fullType) return failure();
    if (!bcastType.hasStaticShape() || !srcType.hasStaticShape())
      return failure();
    if (srcType.getRank() != 1) return failure();
    if (bcastType.getRank() < 2) return failure();

    // Ищем, по какой оси шёл broadcast. Если размерностей несколько
    // подходящих — не трогаем (маловероятный случай).
    int64_t bcastDim = -1;
    for (int64_t i = 0; i < bcastType.getRank(); ++i) {
      if (bcastType.getShape()[i] == srcType.getShape()[0]) {
        if (bcastDim != -1) return failure();
        bcastDim = i;
      }
    }
    if (bcastDim < 0) return failure();

    // Строим linalg.generic: ins(%full, %src) outs(%empty)
    Location loc = bcast.getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(
        loc, bcastType.getShape(), bcastType.getElementType());

    int64_t rank = bcastType.getRank();
    SmallVector<AffineExpr> fullId, srcId;
    for (int64_t i = 0; i < rank; ++i) {
      fullId.push_back(rewriter.getAffineDimExpr(i));
      if (i == bcastDim) srcId.push_back(rewriter.getAffineDimExpr(i));
    }
    auto fullMap = AffineMap::get(rank, 0, fullId, rewriter.getContext());
    auto srcMap  = AffineMap::get(rank, 0, srcId,  rewriter.getContext());

    SmallVector<AffineMap> maps = {fullMap, srcMap, fullMap};
    SmallVector<utils::IteratorType> iters(rank,
                                            utils::IteratorType::parallel);

    Value full = fullOperand;
    Value src  = bcast.getInput();
    bool isRhs = bcastIsRhs;

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{bcastType}, ValueRange{full, src}, ValueRange{empty},
        maps, iters,
        [isRhs](OpBuilder &b, Location loc, ValueRange args) {
          Value lhs = isRhs ? args[0] : args[1];
          Value rhs = isRhs ? args[1] : args[0];
          Value v = b.create<ArithOp>(loc, lhs, rhs);
          b.create<linalg::YieldOp>(loc, v);
        });

    rewriter.replaceOp(user, genericOp.getResults());
    rewriter.eraseOp(bcast);
    return success();
  }
};

struct FuseBroadcastIntoArithPass
    : public PassWrapper<FuseBroadcastIntoArithPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseBroadcastIntoArithPass)

  StringRef getArgument() const override {
    return "fuse-broadcast-into-arith";
  }
  StringRef getDescription() const override {
    return "Fuse linalg.broadcast into arith.subf/mulf/addf to remove "
           "full-size broadcast temp_buffer";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseBroadcastIntoArith<arith::SubFOp>>(&getContext());
    patterns.add<FuseBroadcastIntoArith<arith::MulFOp>>(&getContext());
    patterns.add<FuseBroadcastIntoArith<arith::AddFOp>>(&getContext());
    // applyPatternsGreedily — актуальный API в LLVM 21+, в отличие от
    // устаревшего applyPatternsAndFoldGreedily.
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createFuseBroadcastIntoArithPass() {
  return std::make_unique<FuseBroadcastIntoArithPass>();
}

void registerFuseBroadcastIntoArithPass() {
  PassRegistration<FuseBroadcastIntoArithPass>();
}
