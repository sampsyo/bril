//===- BrilPasses.td - Bril dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BRIL_PASS
#define BRIL_PASS

include "mlir/Pass/PassBase.td"

def ConvertBrilToStd: Pass<"convert-bril-to-std", "::mlir::ModuleOp"> {
  let summary = "Convert Bril dialect to Standard dialect.";
  let description = [{
    This pass converts operations in the Bril dialect to equivalent
    operations in the Standard dialect.
  }];
  let dependentDialects = ["LLVM::LLVMDialect", "arith::ArithDialect", "func::FuncDialect", "cf::ControlFlowDialect"];
}

def RenameMainFunction: Pass<"rename-main-function", "::mlir::ModuleOp"> {
  let summary = "Rename 'main' function to avoid name clashes.";
  let description = [{
    This pass renames the 'main' function to prevent name clashes during linking.
  }];
}

#endif // BRIL_PASS
