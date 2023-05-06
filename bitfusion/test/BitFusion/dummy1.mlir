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
        bitfusion.setup (%0, %1) : i8, i8
        bitfusion.blockend (%3) : i32
        
        return
    }

    func.func @matmul() {
        %0 = arith.constant 2 : i8
        %1 = arith.constant 4 : i8
        %3 = arith.constant 64 : i32
        %num_words = arith.constant 64 : i16
        %lower_bound = arith.constant 64 : i16
        %type_wt = arith.constant 0 : i8
        %type_ip = arith.constant 1 : i8
        %type_op = arith.constant 2 : i8
        %zero = arith.constant 0 : i8   //mul
        %one = arith.constant 1 : i8    //add
        %two = arith.constant 2 : i8    //max
        %mem_width = arith.constant 64 : i32

        bitfusion.setup (%0, %1) : i8, i8
        %ip = bitfusion.ldmem (%type_ip, %zero, %mem_width, %num_words) : i8, i8, i32, i16
        %wt = bitfusion.ldmem (%type_wt, %zero, %mem_width, %num_words) : i8, i8, i32, i16
        %op = bitfusion.ldmem (%type_op, %zero, %mem_width, %num_words) : i8, i8, i32, i16

        bitfusion.loop (%zero, %zero, %lower_bound to %num_words) : i8, i8, i16, i16
            %ip_buf = bitfusion.rdbuf (%type_ip, %zero) : i8, i8
            %wt_buf = bitfusion.rdbuf (%type_wt, %zero) : i8, i8
            %op_buf = bitfusion.rdbuf (%type_op, %zero) : i8, i8
            %temp = bitfusion.compute (%zero, %zero, %ip_buf, %wt_buf) : i8, i8, i16, i16
            %temp2 = bitfusion.compute (%zero, %zero, %temp, %op_buf) : i8, i8, i16, i16
            bitfusion.wrbuf (%type_op, %zero, %temp2) : i8, i8, i16

        bitfusion.stmem (%type_op, %zero, %mem_width, %num_words) : i8, i8, i32, i16 
        bitfusion.blockend (%3) : i32
        
        return
    }
}
