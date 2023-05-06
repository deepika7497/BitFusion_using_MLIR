// RUN: bitfusion-opt %s | bitfusion-opt | FileCheck %s

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
}
