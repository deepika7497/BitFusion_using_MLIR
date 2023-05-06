add_mlir_dialect(BitFusionOps bitfusion)
add_mlir_doc(BitFusionDialect BitFusionDialect BitFusion/ -gen-dialect-doc)
add_mlir_doc(BitFusionOps BitFusionOps BitFusion/ -gen-op-doc)

add_subdirectory(Conversion)
