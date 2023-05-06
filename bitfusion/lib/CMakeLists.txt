add_subdirectory(CAPI)
add_subdirectory(BitFusion)
add_subdirectory(Conversion)

add_mlir_library(BitFusionMLIRInitAll
  InitAll.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRAffineToBitFusion
)