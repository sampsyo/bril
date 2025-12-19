//===- BrilDialect.td - Bril dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BRIL_DIALECT
#define BRIL_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Bril dialect definition.
//===----------------------------------------------------------------------===//

def Bril_Dialect : Dialect {
    let name = "bril";
    let summary = "An MLIR dialect for Bril.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::bril";
    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base bril operation definition.
//===----------------------------------------------------------------------===//

class Bril_Op<string mnemonic, list<Trait> traits = []> :
        Op<Bril_Dialect, mnemonic, traits>;
#endif // BRIL_DIALECT