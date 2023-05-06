// RUN: bitfusion-opt %s | bitfusion-opt | FileCheck %s
// Checks if the bitfusion operations are recognized properly
module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = bitfusion.foo %{{.*}} : i32
        %res = bitfusion.foo %0 : i32
        return
    }

    // CHECK-LABEL: func @bitfusion_types(%arg0: !bitfusion.custom<"10">)
    func.func @bitfusion_types(%arg0: !bitfusion.custom<"10">) {
        return
    }

    func.func @setup_end() {
        %0 = arith.constant 2 : i8
        %1 = arith.constant 4 : i8
        %3 = arith.constant 64 : i32
        bitfusion.setup (%0) : i8
        bitfusion.blockend (%3) : i32
        
        return
    }
}
