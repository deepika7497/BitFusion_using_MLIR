add_mlir_dialect_library(MLIRBitFusion
        BitFusionOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/BitFusion

        DEPENDS
        MLIRBitFusionOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
	)
