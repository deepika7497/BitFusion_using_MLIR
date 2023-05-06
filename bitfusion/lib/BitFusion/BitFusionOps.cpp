//===- BitFusionOps.cpp - BitFusion dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "BitFusion/BitFusionOps.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::bitfusion;

//===----------------------------------------------------------------------===//
// BitFusion dialect initialize
//===----------------------------------------------------------------------===//

void BitFusionDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "BitFusion/BitFusionOps.cpp.inc"
      >();
  registerTypes();
}

void BitFusionDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "BitFusion/BitFusionOpsTypes.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "BitFusion/BitFusionOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "BitFusion/BitFusionOpsTypes.cpp.inc"

#include "BitFusion/BitFusionOpsDialect.cpp.inc"