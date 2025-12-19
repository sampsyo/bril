//===- BrilPasses.h - Bril passes  ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BRIL_BRILPASSES_H
#define BRIL_BRILPASSES_H

#include "bril/BrilDialect.h"
#include "bril/BrilOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace bril {
#define GEN_PASS_DECL
#include "bril/BrilPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "bril/BrilPasses.h.inc"
} // namespace bril
} // namespace mlir

#endif
