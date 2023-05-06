//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "BitFusion/InitAll.h"
#include "BitFusion/BitFusionOps.h"
#include "BitFusion/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/IR/Dialect.h"
// #include "mlir/Dialect/Tosa/IR/TosaOps.h"

// namespace mlir {
namespace bitfusion {
void registerAllDialects(mlir::DialectRegistry &registry) {
  registry
      .insert<mlir::bitfusion::BitFusionDialect>();
}

void registerAllPasses() {
  // registerCanonicalizer();
  mlir::registerConversionPasses();
}
} // namespace bitfusion
// } // namespace mlir
