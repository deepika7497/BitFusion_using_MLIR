//===- BitFusionOps.h - BitFusion dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BITFUSION_BITFUSIONOPS_H
#define BITFUSION_BITFUSIONOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "BitFusion/BitFusionOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "BitFusion/BitFusionOpsTypes.h.inc"
#define GET_OP_CLASSES
#include "BitFusion/BitFusionOps.h.inc"

#endif // BITFUSION_BITFUSIONOPS_H
