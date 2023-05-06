//===- AffineToBitFusion.h - Convert Affine to Standard dialect --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_AFFINETOBITFUSION_H
#define MLIR_CONVERSION_AFFINETOBITFUSION_H

#include "mlir/Support/LLVM.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class AffineForOp;
class Location;
struct LogicalResult;
class OpBuilder;
class Pass;
class RewritePattern;
class Value;
class ValueRange;

class RewritePatternSet;

#define GEN_PASS_DECL
#include "BitFusion/Conversion/Passes.h.inc"
} // namespace mlir

namespace bitfusion {
/// Lowers affine control flow operations (ForStmt, IfStmt and AffineApplyOp)
/// to equivalent lower-level constructs (flow of basic blocks and arithmetic
/// primitives).
std::unique_ptr<mlir::Pass> createAffinetoBitFusion();

} // namespace bitfusion

#endif // MLIR_CONVERSION_AFFINETOBITFUSION_H
