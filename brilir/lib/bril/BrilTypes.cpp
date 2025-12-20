//===- BrilTypes.cpp - Bril dialect types -----------*- C++ -*-------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bril/BrilTypes.h"

#include "bril/BrilDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::bril;

#define GET_TYPEDEF_CLASSES
#include "bril/BrilOpsTypes.cpp.inc"

void BrilDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "bril/BrilOpsTypes.cpp.inc"
      >();
}
