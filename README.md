// CastChainSimplifyPass.cpp
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

// Паттерн для поиска и свёртки замкнутой цепочки приведений.
struct FoldClosedCastChain : public OpRewritePattern<arith::TruncFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::TruncFOp truncOp,
                                PatternRewriter &rewriter) const override {
    // 1. Проверяем, что результатом truncf является f16 (типичный конец цепочки).
    auto resultType = dyn_cast<FloatType>(truncOp.getType());
    if (!resultType || !resultType.isF16())
      return failure();

    // 2. Прослеживаем цепочку в обратном направлении: truncf <- i8 <- i32 <- f32
    Value current = truncOp.getIn();
    SmallVector<Operation *> chain;

    // Шаг 2a: i8 -> f16 (ищем truncf из i8 в f16)
    auto truncI8ToF16 = current.getDefiningOp<arith::TruncFOp>();
    if (!truncI8ToF16 || !truncI8ToF16.getIn().getType().isInteger(8))
      return failure();
    chain.push_back(truncI8ToF16);
    current = truncI8ToF16.getIn();

    // Шаг 2b: i32 -> i8 (ищем truncf из i32 в i8)
    auto truncI32ToI8 = current.getDefiningOp<arith::TruncIOp>();
    if (!truncI32ToI8 || !truncI32ToI8.getIn().getType().isInteger(32))
      return failure();
    chain.push_back(truncI32ToI8);
    current = truncI32ToI8.getIn();

    // Шаг 2c: f32 -> i32 (ищем fptosi из f32 в i32)
    auto fptosiF32ToI32 = current.getDefiningOp<arith::FPToSIOp>();
    if (!fptosiF32ToI32 || !fptosiF32ToI32.getIn().getType().isF32())
      return failure();
    chain.push_back(fptosiF32ToI32);
    Value originalF32 = fptosiF32ToI32.getIn();

    // 3. Проверка безопасности: цепочка должна быть замкнутой, т.е. исходный
    //    и конечный тип совпадают (f32 -> ... -> f32). В нашем случае это так,
    //    но мы заменяем только конечный truncf, поэтому проверяем, что
    //    originalF32 имеет тип f32.
    if (!originalF32.getType().isF32())
      return failure();

    // 4. Заменяем конечный truncf на исходное значение f32.
    //    Это делает всю цепочку мёртвой, и DCE удалит промежуточные операции.
    rewriter.replaceOp(truncOp, originalF32);
    return success();
  }
};

// Пасс, работающий на уровне ModuleOp.
struct CastChainSimplifyPass
    : public PassWrapper<CastChainSimplifyPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CastChainSimplifyPass)

  StringRef getArgument() const final { return "cast-chain-simplify"; }
  StringRef getDescription() const final {
    return "Fold closed f32->i32->i8->f16->f32 cast chains to reduce UB usage";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    RewritePatternSet patterns(&getContext());
    // Добавляем паттерн. Можно также добавить канонические паттерны для arith.
    patterns.add<FoldClosedCastChain>(&getContext());
    // Применяем паттерны жадно, чтобы цепочки свёртывались до конца.
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
        signalPassFailure();
    }
  }
};

} // namespace

// Фабричная функция для создания пасса.
std::unique_ptr<Pass> createCastChainSimplifyPass() {
  return std::make_unique<CastChainSimplifyPass>();
}
