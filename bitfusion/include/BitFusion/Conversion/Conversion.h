//===- Passes.h - Conversion Pass Construction and Registration -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BITFUSION_CONVERSION_H
#define MLIR_BITFUSION_CONVERSION_H

// #include "BitFusion/Conversion/AffineToBitFusion.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL
#include "BitFusion/Conversion/Passes.h.inc"
// } // namespace mlir

namespace bitfusion {

std::unique_ptr<mlir::Pass> createAffinetoBitFusion();

} // namespace bitfusion
} // namespace mlir


#endif // MLIR_BITFUSION_CONVERSION_H
