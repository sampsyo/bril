//===- MLIRGen.h - MLIR Generation from a Bril JSON
//------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a simple interface to perform IR generation targeting MLIR
// from a JSON representation for the Bril language.
//
//===----------------------------------------------------------------------===//

#ifndef BRIL_MLIRGEN_H
#define BRIL_MLIRGEN_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include <memory>
#include <nlohmann/json.hpp>

namespace bril {
class ModuleAST;

/// Emit IR for the given Bril moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          nlohmann::json &json);
} // namespace bril

#endif // BRIL_MLIRGEN_H
