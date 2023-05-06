//====- AffineToBitfusion.cpp - Partial lowering from Affine+Std to Bitfusion--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of affine loops, memref operations and 
// standard operations to the bitfusion dialect. 
// Not sure: This lowering
// expects that all calls have been inlined, and all shapes have been resolved.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/Sequence.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Utils.h"
// #include "mlir/Conversion/StandardToLLVM/StandardToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// #include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "BitFusion/BitFusionOps.h"
#include "BitFusion/Conversion/Conversion.h"
// #include "BitFusion/Passes.h"

// namespace mlir {
// #define GEN_PASS_DEF_CONVERTAFFINETOBITFUSION
// #include "BitFusion/Passes.h.inc"
// } // namespace mlir

using namespace mlir;
// using namespace mlir::bitfusion;

/// Given a range of values, emit the code that reduces them with "min" or "max"
/// depending on the provided comparison predicate.  The predicate defines which
/// comparison to perform, "lt" for "min", "gt" for "max" and is used for the
/// `cmpi` operation followed by the `select` operation:
///
///   %cond   = arith.cmpi "predicate" %v0, %v1
///   %result = select %cond, %v0, %v1
///
/// Multiple values are scanned in a linear sequence.  This creates a data
/// dependences that wouldn't exist in a tree reduction, but is easier to
/// recognize as a reduction by the subsequent passes.
static Value buildMinMaxReductionSeq(Location loc,
                                     arith::CmpIPredicate predicate,
                                     ValueRange values, OpBuilder &builder) {
  assert(!values.empty() && "empty min/max chain");

  auto valueIt = values.begin();
  Value value = *valueIt++;
  for (; valueIt != values.end(); ++valueIt) {
    auto cmpOp = builder.create<arith::CmpIOp>(loc, predicate, value, *valueIt);
    value = builder.create<arith::SelectOp>(loc, cmpOp.getResult(), value,
                                            *valueIt);
  }

  return value;
}

/// Emit instructions that correspond to computing the maximum value among the
/// values of a (potentially) multi-output affine map applied to `operands`.
static Value lowerAffineMapMax(OpBuilder &builder, Location loc, AffineMap map,
                               ValueRange operands) {
  if (auto values = expandAffineMap(builder, loc, map, operands))
    return buildMinMaxReductionSeq(loc, arith::CmpIPredicate::sgt, *values,
                                   builder);
  return nullptr;
}

/// Emit instructions that correspond to computing the minimum value among the
/// values of a (potentially) multi-output affine map applied to `operands`.
static Value lowerAffineMapMin(OpBuilder &builder, Location loc, AffineMap map,
                               ValueRange operands) {
  if (auto values = expandAffineMap(builder, loc, map, operands))
    return buildMinMaxReductionSeq(loc, arith::CmpIPredicate::slt, *values,
                                   builder);
  return nullptr;
}

/// Emit instructions that correspond to the affine map in the upper bound
/// applied to the respective operands, and compute the minimum value across
/// the results.
Value mlir::lowerAffineUpperBound(AffineForOp op, OpBuilder &builder) {
  return lowerAffineMapMin(builder, op.getLoc(), op.getUpperBoundMap(),
                           op.getUpperBoundOperands());
}

/// Emit instructions that correspond to the affine map in the lower bound
/// applied to the respective operands, and compute the maximum value across
/// the results.
Value mlir::lowerAffineLowerBound(AffineForOp op, OpBuilder &builder) {
  return lowerAffineMapMax(builder, op.getLoc(), op.getLowerBoundMap(),
                           op.getLowerBoundOperands());
}

namespace {
//===----------------------------------------------------------------------===//
// AffineToBitfusion RewritePatterns: AffineFor operations
//===----------------------------------------------------------------------===//

class AffineForLowering : public OpRewritePattern<AffineForOp> {
public:
  using OpRewritePattern<AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineForOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lowerBound = lowerAffineLowerBound(op, rewriter);
    Value upperBound = lowerAffineUpperBound(op, rewriter);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, op.getStep());
    auto scfForOp = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound,
                                                step, op.getIterOperands());
    rewriter.eraseBlock(scfForOp.getBody());
    rewriter.inlineRegionBefore(op.getRegion(), scfForOp.getRegion(),
                                scfForOp.getRegion().end());
    rewriter.replaceOp(op, scfForOp.getResults());
    return success();
  }
};

// class AffineForLowering : public OpRewritePattern<AffineForOp> {
// public:
//   using OpRewritePattern<AffineForOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(AffineForOp op,
//                                 PatternRewriter &rewriter) const override {
//     Location loc = op.getLoc();
//     Value lowerBound = lowerAffineLowerBound(op, rewriter);
//     Value upperBound = lowerAffineUpperBound(op, rewriter);
//     Value step = rewriter.create<arith::ConstantIndexOp>(loc, op.getStep());
//     auto scfForOp = rewriter.create<bitfusion::LoopOp>(loc, lowerBound, upperBound,
//                                                 step, op.getIterOperands());
//     rewriter.eraseBlock(scfForOp.getBody());
//     rewriter.inlineRegionBefore(op.getRegion(), scfForOp.getRegion(),
//                                 scfForOp.getRegion().end());
//     rewriter.replaceOp(op, scfForOp.getResults());
//     return success();
//   }
// };

// struct AffineForOpLowering : public ConversionPattern {
//   AffineForOpLowering(MLIRContext *ctx)
//       : ConversionPattern(mlir::AffineForOp::getOperationName(), 1, ctx) {}

//   LogicalResult
//   matchAndRewrite(Operation *op, ArrayRef<Value> operands,
//                   ConversionPatternRewriter &rewriter) const final {
    
//     // Get loop bounds and step value.
//     auto lb = op.getLowerBound();
//     auto ub = op.getUpperBound();
//     auto step = op.getStep();

//     // Create new loop operation.
//     auto loc = op.getLoc();
//     auto loop = rewriter.create<bitfusion::LoopOp>(loc, TypeRange(), TypeRange(), lb, ub, step);

//     // Add loop level, loop-id and num-iterations attributes to the loop operation.
//     int8_t loopLevel = 0; // Set to appropriate value based on the loop nesting level.
//     int8_t loopId = 0; // Set to appropriate value based on the loop nesting level and position.
//     int16_t numIterations = ub; // Set to appropriate value based on the loop bounds and step.
//     loop.setAttr("loop_level", rewriter.getI64IntegerAttr(loopLevel));
//     loop.setAttr("loop_id", rewriter.getI64IntegerAttr(loopId));
//     loop.setAttr("num_iterations", rewriter.getI64IntegerAttr(numIterations));

//     // Replace the original affine.for with the new loop operation.
//     rewriter.replaceOp(op, loop.getResult(0));
    
//     return success();
//   }
// };

// ===----------------------------------------------------------------------===//
// AffineToBitfusion RewritePatterns: AffineLoad operations
// ===----------------------------------------------------------------------===//

// struct AffineLoadOpLowering : public ConversionPattern {
//   AffineLoadOpLowering(MLIRContext *ctx)
//       : ConversionPattern(mlir::AffineLoadOp::getOperationName(), 1, ctx) {}

//   LogicalResult
//   matchAndRewrite(Operation *op, ArrayRef<Value> operands,
//                   ConversionPatternRewriter &rewriter) const final {
    
//     // Get load map and memref typr.
//     auto map = op.getAffineMap();
//     auto memrefType = op.getMemRefType();

//     // Create new rd_buf operation.
//     auto loc = op->getLoc();
//     auto rd_buf = rewriter.create<bitfusion::RdBufOp >(loc, MemRefType, op.memref());

//     // Add spad-type, loop-id and num-iterations attributes to the rd_buf operation.
//     int8_t spadType = 0; // Set to appropriate value based on the rd_buf nesting level.
//     int8_t loopId = 0; // Set to appropriate value based on the rd_buf nesting level and position.
//     rd_buf.setAttr("spad_type", rewriter.getI64IntegerAttr(spadType));
//     rd_buf.setAttr("loop_id", rewriter.getI64IntegerAttr(loopId));

//     // Replace the original affine.for with the new loop operation.
//     rewriter.replaceOp(op, rd_buf.getResult(0));
    
//     return success();
//   }
// };

// ===----------------------------------------------------------------------===//
// AffineToBitfusion RewritePatterns: AffineStore operations
// ===----------------------------------------------------------------------===//

// struct AffineStoreOpLowering : public ConversionPattern {
//   AffineStoreOpLowering(MLIRContext *ctx)
//       : ConversionPattern(mlir::AffineStoreOp::getOperationName(), 1, ctx) {}

//   LogicalResult
//   matchAndRewrite(Operation *op, ArrayRef<Value> operands,
//                   ConversionPatternRewriter &rewriter) const final {
    
//     // Get store map and memref typr.
//     auto map = op.getAffineMap();
//     auto memrefType = op.getMemRefType();

//     // Create new rd_buf operation.
//     auto loc = op.getLoc();
//     auto wr_buf = rewriter.create<bitfusion::WrBufOp>(loc, MemRefType, op.memref());

//     // Add spad-type, loop-id and num-iterations attributes to the rd_buf operation.
//     int8_t spadType = 0; // Set to appropriate value based on the rd_buf nesting level.
//     int8_t loopId = 0; // Set to appropriate value based on the rd_buf nesting level and position.
//     wr_buf.setAttr("spad_type", rewriter.getI64IntegerAttr(spadType));
//     wr_buf.setAttr("loop_id", rewriter.getI64IntegerAttr(loopId));

//     // Replace the original affine.for with the new loop operation.
//     rewriter.replaceOp(op, wr_buf.getResult(0));
    
//     return success();
//   }
// };

} //namespace

//===----------------------------------------------------------------------===//
// ConvertAffineToBitFusion
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops to the bitfusion dialect
namespace {
struct ConvertAffineToBitFusion
    : public PassWrapper<ConvertAffineToBitFusion, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertAffineToBitFusion)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, func::FuncDialect, memref::MemRefDialect,
                    mlir::bitfusion::BitFusionDialect >();
  }
  void runOnOperation() final;
};
} //namespace

void ConvertAffineToBitFusion::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine`, `Arith`, `Func`, and `MemRef` dialects.
  target.addLegalDialect<BuiltinDialect, func::FuncDialect , memref::MemRefDialect, 
                          arith::ArithDialect, mlir::bitfusion::BitFusionDialect>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the mlir operations.
  RewritePatternSet patterns(&getContext());
  patterns.add<AffineForLowering>(
      &getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

namespace bitfusion {
/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// to a subset of the Bitfusion IR (e.g. matmul).
std::unique_ptr<Pass> createAffinetoBitFusion() {
  return std::make_unique<ConvertAffineToBitFusion>();
}

} //namespace bitfusion