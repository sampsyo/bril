#include "bril/BrilDialect.h"
#include "bril/MLIRGen.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <nlohmann/json.hpp>
#include <system_error>

int main() {
  mlir::MLIRContext context;

  context.getOrLoadDialect<mlir::bril::BrilDialect>();

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getSTDIN();
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Error reading from stdin: " << ec.message() << "\n";
    return 1;
  }

  auto buffer = fileOrErr->get()->getBuffer();

  nlohmann::json brilJson = nlohmann::json::parse(buffer);

  auto module = bril::mlirGen(context, brilJson);

  if (!module) {
    llvm::errs() << "Failed to generate MLIR module\n";
    return 1;
  }

  module->dump();

  return 0;
}