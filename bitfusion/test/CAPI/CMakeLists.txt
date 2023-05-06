# Here we create a single aggregate shared library with the parts of the CAPI
# that we want to bundle together. Then we link a simple C executable
# against it to demonstrate that it does have the fully self contained
# core MLIR library and our own bitfusion dialect.
add_mlir_aggregate(BitFusionCAPITestLib
  SHARED
  EMBED_LIBS
  MLIRCAPIIR
  # TODO: Remove this in favor of showing fine grained dialect registration
  # (once available).
  MLIRCAPIRegisterEverything
  BitFusionCAPI
)

add_llvm_executable(bitfusion-capi-test
  bitfusion-capi-test.c
)
llvm_update_compile_flags(bitfusion-capi-test)
target_link_libraries(bitfusion-capi-test
  PRIVATE BitFusionCAPITestLib)
