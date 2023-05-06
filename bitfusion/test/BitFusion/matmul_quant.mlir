// RUN: bitfusion-opt %s -convert-linalg-to-affine-loops | FileCheck %s

// Test that we can lower all the way to LLVM without crashing, don't check results here.
// RUN: bitfusion-opt %s -convert-linalg-to-affine-loops -convert-linalg-to-llvm='use-opaque-pointers=1' -o=/dev/null 2>&1

func.func @matmul(%arg0: memref<?xi8>, %M: index, %N: index, %K: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %A = memref.view %arg0[%c0][%M, %K] : memref<?xi8> to memref<?x?xi8>
  %B = memref.view %arg0[%c0][%K, %N] : memref<?xi8> to memref<?x?xi4>
  %C = memref.view %arg0[%c0][%M, %N] : memref<?xi8> to memref<?x?xi8>
  linalg.matmul ins(%A, %B: memref<?x?xi8>, memref<?x?xi4>)
               outs(%C: memref<?x?xi8>)
  return
}

//----------------------------------------------------------------------------//
// Named ops to loops.
//----------------------------------------------------------------------------//
func.func @named_batch_matmul(%A: memref<?x?x?xi8>, %B: memref<?x?x?xi4>, %C: memref<?x?x?xi8>) {
  linalg.batch_matmul ins(%A, %B: memref<?x?x?xi8>, memref<?x?x?xi4>)
                     outs(%C : memref<?x?x?xi8>)
  return
}
// CHECK-LABEL: @named_batch_matmul
//  CHECK-SAME: %[[mA:[a-zA-Z0-9]+]]: memref<?x?x?xi8>
//  CHECK-SAME: %[[mB:[a-zA-Z0-9]+]]: memref<?x?x?xi4>
//  CHECK-SAME: %[[mC:[a-zA-Z0-9]+]]: memref<?x?x?xi8>
//       CHECK: %[[B:.*]] = memref.dim %[[mA]], %c0 : memref<?x?x?xi8>
//       CHECK: %[[M:.*]] = memref.dim %[[mA]], %c1 : memref<?x?x?xi8>
//       CHECK: %[[K:.*]] = memref.dim %[[mA]], %c2 : memref<?x?x?xi8>
//       CHECK: %[[N:.*]] = memref.dim %[[mB]], %c2 : memref<?x?x?xi4>
//       CHECK: affine.for %[[b:.*]] = {{.*}}0 to %[[B]] {
//       CHECK:   affine.for %[[m:.*]] = {{.*}}0 to %[[M]] {
//       CHECK:     affine.for %[[n:.*]] = {{.*}}0 to %[[N]] {
//       CHECK:       affine.for %[[k:.*]] = {{.*}}0 to %[[K]] {
//       CHECK:       %[[va:.*]] = affine.load %[[mA]][%[[b]], %[[m]], %[[k]]] : memref<?x?x?xi8>
//       CHECK:       %[[vb:.*]] = affine.load %[[mB]][%[[b]], %[[k]], %[[n]]] : memref<?x?x?xi4>
//       CHECK:       %[[vc:.*]] = affine.load %[[mC]][%[[b]], %[[m]], %[[n]]] : memref<?x?x?xi8>
//       CHECK:       %[[vbb:.*]] = arith.extsi %[[vb]] : i4 to i8
//       CHECK:       %[[inc:.*]] = arith.muli %[[va]], %[[vbb]] : i8
//       CHECK:       %[[res:.*]] = arith.addi %[[vc]], %[[inc]] : i8
//       CHECK:       affine.store %[[res]], %[[mC]][%[[b]], %[[m]], %[[n]]] : memref<?x?x?xi8>
