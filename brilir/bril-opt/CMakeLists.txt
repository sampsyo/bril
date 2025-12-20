set(LIBS
    MLIRArithDialect
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRBril
  )
add_llvm_executable(bril-opt bril-opt.cpp)

llvm_update_compile_flags(bril-opt)
target_link_libraries(bril-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(bril-opt)
