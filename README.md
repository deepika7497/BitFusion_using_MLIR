# BitFusion_using_MLIR
This is the project repository for ECE 663 (Spring 2023) at Purdue University.

To run this example first install llvm and MLIR following the instructions here:
https://mlir.llvm.org/getting_started/

Add the LLVM build directory path to `$LLVM_BUILD_DIR`

The diirectory structure for `bitfusion/` is as follows:
`bitfusion-opt/` -> The bitfusion `opt` tool is created here and the available dialects (works) and passes (nopt working) are registered.
`bitfusion-translate/` -> Not used as I did not register any new translations.
`include/` -> contains the table gen and include files (.h) for the dialect and conversion passes.
`lib/` -> contains the C++ files that define the passes 
`test/` -> Contains the examples used when `check-bitfusion` is run. The examples can be tried out with `$BITFUSION_BUILD_DIR/bitfusion-opt $PATH_TO_FILE`

# An out-of-tree MLIR dialect

This is an example of an out-of-tree [MLIR](https://mlir.llvm.org/) dialect along with a bitfusion `opt`-like tool to operate on that dialect. This is based on the standalone dialect example provided with mlir.

## Building

To build and launch the tests, run
```sh
cd bitfusion
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$LLVM_BUILD_DIR/bin/llvm-lit
cmake --build . --target check-bitfusion
```
Should Pass 7 tests and fail 1.

To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.