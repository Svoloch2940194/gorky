#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h" // Нужно для работы с файловой системой
#include <atomic>

using namespace mlir;

namespace {

class DumpTTAdapterPass : public PassWrapper<DumpTTAdapterPass, OperationPass<ModuleOp>> {
private:
    // Статический счетчик, чтобы различать вызовы пасса в конвейере
    static inline std::atomic<int> callCounter{0};

public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DumpTTAdapterPass)

    StringRef getArgument() const override { return "dump-tt-adapter"; }
    StringRef getDescription() const override { return "Dumps TTAdapter IR to a sequence of files."; }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        int currentCall = ++callCounter;

        std::string filename = "ttadapter_dump_" + std::to_string(currentCall) + ".mlir";
        
        std::error_code ec;
        llvm::raw_fd_ostream fileStream(filename, ec, llvm::sys::fs::OF_Text);
        
        if (ec) {
            llvm::errs() << "[ERROR] Failed to open file for dumping: " << ec.message() << "\n";
            return;
        }

        fileStream << "=============================================\n";
        fileStream << "=== [Full IR Dump #" << currentCall << "] ===\n";
        fileStream << "=============================================\n\n";

        // Печатаем ВЕСЬ модуль целиком со всеми операциями
        module->print(fileStream);

        fileStream.flush();
    }
};

} // namespace

// Функция-фабрика для создания пасса
std::unique_ptr<Pass> createDumpTTAdapterPass() {
    return std::make_unique<DumpTTAdapterPass>();
}
