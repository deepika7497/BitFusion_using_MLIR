//===- Passes.h - Conversion Pass Construction and Registration -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BITFUSION_CONVERSION_PASSES_H
#define MLIR_BITFUSION_CONVERSION_PASSES_H

#include "BitFusion/Conversion/Conversion.h"

namespace mlir {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "BitFusion/Conversion/Passes.h.inc"

} // namespace mlir

#endif // MLIR_BITFUSION_CONVERSION_PASSES_H
