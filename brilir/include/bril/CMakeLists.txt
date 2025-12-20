add_mlir_dialect(BrilOps bril)

set(LLVM_TARGET_DEFINITIONS BrilPasses.td)
mlir_tablegen(BrilPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRBrilPassesIncGen)