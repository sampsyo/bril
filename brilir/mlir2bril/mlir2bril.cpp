#include "bril/MLIR2Bril.h"
#include "bril/BrilDialect.h"
#include <iostream>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

int main() {
  mlir::MLIRContext context;

  context.getOrLoadDialect<mlir::bril::BrilDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getSTDIN();
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Error reading from stdin: " << ec.message() << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error parsing MLIR module from stdin\n";
    return 1;
  }

  nlohmann::json brilJson = bril::mlirToBril(*module);

  std::cout << brilJson.dump(2);

  return 0;
}