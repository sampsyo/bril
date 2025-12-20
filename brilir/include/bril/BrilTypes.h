//===- BrilTypes.h - Bril dialect types -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BRIL_BRILTYPES_H
#define BRIL_BRILTYPES_H

#include "mlir/IR/BuiltinTypes.h"

#define GET_TYPEDEF_CLASSES
#include "bril/BrilOpsTypes.h.inc"

#endif // BRIL_BRILTYPES_H
