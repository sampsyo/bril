# An out-of-tree MLIR dialect for Bril

This repo contains an out-of-tree [MLIR](https://mlir.llvm.org/) dialect for [Bril](https://capra.cs.cornell.edu/bril/intro.html), a standalone `opt`-like tool to operate on Bril dialect, and conversion tools for translating between Bril and MLIR (`bril2mlir` and `mlir2bril`).

## Building

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`.
Ensure that all required dependencies listed [here](https://mlir.llvm.org/getting_started/) and [nlohmann-json](https://github.com/nlohmann/json) are installed; using `CCache` is strongly recommended.

```sh
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build
cd build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON \
   -DLLVM_CCACHE_BUILD=ON \
   -DCMAKE_INSTALL_PREFIX=$HOME/opt/llvm \
   -DLLVM_INSTALL_UTILS=ON
cmake --build . --target install
```

To build `brilir` (ensure `BUILD_DIR` and `PREFIX` match the paths used in the previous step).

```sh
export BUILD_DIR=$HOME/path/to/llvm-project/build
export PREFIX=$HOME/opt/llvm
mkdir build
cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DLLVM_CCACHE_BUILD=ON
cmake --build .
```

## Example Usage

`bril2mlir` expects the Bril source to be in SSA form, which can be prepared as follows (using the [example implementations](https://github.com/sampsyo/bril/tree/main/examples) from the Bril repo):

```sh
bril2json < source.bril | python ~/path/to/bril/examples/to_ssa.py | python ~/path/to/bril/examples/tdce.py tdce+
```

For Bril to MLIR conversion:

```sh
bril2mlir < bril_input.json 2>&1 | mlir2bril | bril2txt
```

For lowering to LLVM IR and subsequent linking:

```sh
bril2mlir < bril_input.json 2>&1 | bril-opt --pass-pipeline="builtin.module(convert-bril-to-std,rename-main-function,convert-arith-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,canonicalize,cse)" - | mlir-translate --mlir-to-llvmir - -o output.ll
# The Bril entry point is exposed as `bril_main` and can be linked against.
clang++ output.ll main.cpp
```

All the built tools are located in `build/bin` directory.
