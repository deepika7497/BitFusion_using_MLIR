add_mlir_conversion_library(MLIRAffineToBitFusion
  AffineToBitFusion.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/BitFusion

  DEPENDS
  MLIRBitFusionConversionPassIncGen

  # LINK_COMPONENTS
  # Core

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineUtils
  MLIRArithDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRPass
  MLIRTransforms
  MLIRVectorDialect
  MLIRBitFusion
  )