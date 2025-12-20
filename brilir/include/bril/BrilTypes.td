//===- BrilTypes.td - Bril dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BRIL_TYPES
#define BRIL_TYPES

include "mlir/IR/AttrTypeBase.td"
include "bril/BrilDialect.td"

//===----------------------------------------------------------------------===//
// Bril type definitions
//===----------------------------------------------------------------------===//

class Bril_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Bril_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Ptr : Bril_Type<"Ptr", "ptr"> {
    let summary = "Bril pointer type";
    let description = "Pointer type in Bril dialect";
    let parameters = (ins "Type":$pointeeType);
    let assemblyFormat = "`<` $pointeeType `>`";
}

#endif // BRIL_TYPES
