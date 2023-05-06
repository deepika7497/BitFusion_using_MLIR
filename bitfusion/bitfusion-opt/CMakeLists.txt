get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIROptLib
        MLIRBitFusion
        BitFusionMLIRInitAll
        )
add_llvm_executable(bitfusion-opt bitfusion-opt.cpp)

llvm_update_compile_flags(bitfusion-opt)
target_link_libraries(bitfusion-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(bitfusion-opt)
