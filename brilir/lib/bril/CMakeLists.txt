add_mlir_dialect_library(MLIRBril
        BrilOps.cpp
        BrilDialect.cpp
        BrilPasses.cpp
        BrilTypes.cpp
        MLIRGen.cpp
        MLIR2Bril.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/bril

        DEPENDS
        MLIRBrilOpsIncGen
        MLIRBrilPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
