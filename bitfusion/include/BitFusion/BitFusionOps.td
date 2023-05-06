//===- BitFusionOps.td - BitFusion dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BITFUSION_OPS
#define BITFUSION_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// BitFusion dialect definition.
//===----------------------------------------------------------------------===//

def BitFusion_Dialect : Dialect {
    let name = "bitfusion";
    let summary = "A bitfusion out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::bitfusion";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// BitFusion type definitions
//===----------------------------------------------------------------------===//

class BitFusion_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<BitFusion_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def BitFusion_CustomType : BitFusion_Type<"Custom", "custom"> {
    let summary = "BitFusion custom type";
    let description = "Custom type in bitfusion dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Base bitfusion operation definition.
//===----------------------------------------------------------------------===//

class BitFusion_Op<string mnemonic, list<Trait> traits = []> :
        Op<BitFusion_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// BitFusion Operations.
//===----------------------------------------------------------------------===//

def BitFusion_FooOp : BitFusion_Op<"foo", [Pure,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `bitfusion.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the foo operation to %0
        %1 = bitfusion.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def BitFusion_SetupOp : BitFusion_Op<"setup"> {
    let summary = "Bit Fusion setup operation.";
    let description = [{
        The setup instruction marks the beginning of an instruction block 
        and configures the Fusion Units and its data delivery logic to 
        the specified bitwidth for the operands.

        Fields:
            op0.bitwidth : 2, 4, 8, 16
            op1.bitwidth : 2, 4, 8, 16
    }];

    let arguments = (ins 
        I8:$op0_bitwidth,
        I8:$op1_bitwidth
    );

    let assemblyFormat = [{
        attr-dict `(` $op0_bitwidth `,` $op1_bitwidth `)` `:` type($op0_bitwidth) `,` type($op1_bitwidth)
    }];
}

def BitFusion_BlockEndOp : BitFusion_Op<"blockend"> {
    let summary = "Bit Fusion block-end operation.";
    let description = [{
        The block-end instruction signifies the end of a block and provides 
        the address to the next instruction in the next-inst field.

        Fields:
            Next-addr
    }];

    let arguments = (ins 
        I32:$next_addr
    );

    let assemblyFormat = [{
        attr-dict `(` $next_addr `)` `:` type($next_addr)
    }];
}

def BitFusion_LdMemOp : BitFusion_Op<"ldmem"> {
    let summary = "Bit Fusion ld-mem operation.";
    let description = [{
        The ld-mem/st-mem instructions exchange data between the on-chip buffers 
        and the off-chip memory.

        Fields:
            scratchpad-type
            mem.bitwidth
            loop-id
            num-words
    }];

    let arguments = (ins 
        I8:$spad_type,
        I32:$mem_bitwidth,
        I8:$loop_id,
        I16:$num_words
    );

    let results = (outs I32:$res);

    let assemblyFormat = [{
        attr-dict `(` $spad_type `,` $loop_id `,` $mem_bitwidth `,` $num_words `)` `:` type($spad_type) `,` type($loop_id) `,` type($mem_bitwidth) `,` type($num_words)
    }];
}

def BitFusion_StMemOp : BitFusion_Op<"stmem"> {
    let summary = "Bit Fusion st-mem operation.";
    let description = [{
        The ld-mem/stmem instructions exchange data between the on-chip buffers 
        and the off-chip memory.

        Fields:
            scratchpad-type
            mem.bitwidth
            loop-id
            num-words
    }];

    let arguments = (ins 
        I8:$spad_type,
        I32:$mem_bitwidth,
        I8:$loop_id,
        I16:$num_words
    );

    // let results = (outs I32:$res);

    let assemblyFormat = [{
        attr-dict `(` $spad_type `,` $loop_id `,` $mem_bitwidth `,` $num_words `)` `:` type($spad_type) `,` type($loop_id) `,` type($mem_bitwidth) `,` type($num_words)
    }];
}

def BitFusion_RdBufOp : BitFusion_Op<"rdbuf"> {
    let summary = "Bit Fusion rd-buf operation.";
    let description = [{
        The rd-buf/wr-buf instructions read/write data from the on-chip 
        buffers specified by the scratchpad-type.

        Fields:
            scratchpad-type
            loop-id
    }];

    let arguments = (ins 
        I8:$spad_type,
        I8:$loop_id
    );

    let results = (outs I16:$res);

    let assemblyFormat = [{
        attr-dict `(` $spad_type `,` $loop_id `)` `:` type($spad_type) `,` type($loop_id)
    }];
}

def BitFusion_WrBufOp : BitFusion_Op<"wrbuf"> {
    let summary = "Bit Fusion wr-buf operation.";
    let description = [{
        the rd-buf/wr-buf instructions read/write data from the on-chip 
        buffers specified by the scratchpad-type.

        Fields:
            scratchpad-type
            loop-id
    }];

    let arguments = (ins 
        I8:$spad_type,
        I8:$loop_id,
        I16:$data
    );

    // let results = (outs I32:$res);

    let assemblyFormat = [{
        attr-dict `(` $spad_type `,` $loop_id `,` $data `)` `:` type($spad_type) `,` type($loop_id) `,` type($data)
    }];
}

def BitFusion_GenAddrOp : BitFusion_Op<"genaddr"> {
    let summary = "Bit Fusion gen-addr operation.";
    let description = [{
        The gen-addr instruction dictates how to walk through the elements 
        of the input/output hyper-dimensional arrays. The stride field in 
        the gen-addr instruction specifies how to walk through the array 
        elements in the loop, which is identified by the loop-id field. 
        The words after the setup instruction define the memory base address 
        for the data that fills the three buffers of input, output, and weights. 
        The gen-addr instruction generates the addresses that walk through 
        the memory data and fill the buffers.
        
        Fields:
            scratchpad-type
            ld/st
            loop-id
            stride
    }];

    let arguments = (ins 
        I8:$spad_type,
        I32:$ld_st,
        I8:$loop_id,
        I16:$stride
    );

    let results = (outs I32:$res);

    let assemblyFormat = [{
        attr-dict `(` $spad_type `:` $ld_st `:` $loop_id `:` $stride `)`
    }];
}

def BitFusion_ComputeOp : BitFusion_Op<"compute"> {
    let summary = "Bit Fusion compute operation.";
    let description = [{
        The compute instruction specifies the type of operation.
        MAC/max etc.

        Fields:
            fn
            loop-id
    }];

    let arguments = (ins 
        I8:$fn,
        I8:$loop_id,
        I16:$op1,
        I16:$op2
    );

    let results = (outs I16:$res);

    let assemblyFormat = [{
        attr-dict `(` $fn `,` $loop_id `,` $op1 `,` $op2 `)` `:` type($fn) `,` type($loop_id) `,` type($op1) `,` type($op2)
    }];
}

def BitFusion_LoopOp : BitFusion_Op<"loop"> {
    let summary = "Bit Fusion loop operation.";
    let description = [{
        loop instructions enable a concise way of defining the walks and 
        operations in a DNN layer. Each loop instruction has a unique ID 
        in the block. As shown in Table I, the num-iterations field in 
        the loop instruction defines iteration count.

        Fields:
            loop-level
            loop-id
            num_iterations
    }];

    let arguments = (ins 
        I8:$loop_level,
        I8:$loop_id,
        I16:$lower_bound,
        I16:$num_iterations
    );

    let results = (outs I32:$res);

    let assemblyFormat = [{
        attr-dict `(` $loop_level `,` $loop_id `,` $lower_bound `to` $num_iterations `)` `:` type($loop_level) `,` type($loop_id) `,` type($lower_bound) `,` type($num_iterations)
    }];
}

#endif // BITFUSION_OPS
