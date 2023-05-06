//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BITFUSION_CONVERSION_PASSES
#define MLIR_BITFUSION_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// AffineToBitFusion
//===----------------------------------------------------------------------===//

def ConvertAffineToBitFusion : Pass<"convert-affine-to-bitfusion"> {
  let summary = "Lower Affine operations to a combination of BitFusion and other "
                "operations";
  let description = [{

    Convert operations from the affine dialect into operations from
    bitfusion dialects.

    `affine.for` operations are converted to `bitfusion.for` operations that are free
    of certain structural restrictions (on their bounds and step). `affine.load`
    is similarly converted to the `bitfusion.rd_buf` operation. Consequently, named maps
    and sets thare are no longer in use may be removed from the module.
 
    -   Functions without a body are not modified.
    -   The semantics of the other functions is preserved.
    -   Individual operations other than those mentioned above are not modified
  }];
  let constructor = "mlir::bitfusion::createAffinetoBitFusion()";
  let dependentDialects = [
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "vector::VectorDialect",
    "mlir::bitfusion::BitFusionDialect"
  ];
}

#endif // MLIR_BITFUSION_CONVERSION_PASSES
