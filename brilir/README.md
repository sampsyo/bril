# An out-of-tree MLIR dialect for Bril

This repo contains an out-of-tree [MLIR](https://mlir.llvm.org/) dialect for [Bril](https://capra.cs.cornell.edu/bril/intro.html), a standalone `opt`-like tool to operate on Bril dialect, and conversion tools for translating between Bril and MLIR (`bril2mlir` and `mlir2bril`).

## Building

The first step is to obtain and build MLIR itself.
Ensure you have all the required dependencies listed [in the MLIR documentation](https://mlir.llvm.org/getting_started/).
Then do something like this:

Use something like this to obtain and build the MLIR dependency:

```sh
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build
cd build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON
ninja
```

You may need to change some of the LLVM build options depending on the setup you want.
For example, use `-DLLVM_CCACHE_BUILD=ON` to enable [Ccache](https://ccache.dev) to speed up the build.
This project was most recently tested against commit `598657158b5cb42c07bee949f193d3c8d79ce20f` of the LLVM monorepo.

To build this Bril dialect, you'll need the [nlohmann-json](https://github.com/nlohmann/json) library.
Using [Homebrew](https://brew.sh), for instance, you can type `brew install nlohmann-json`.

To build `brilir` (ensure `BUILD_DIR` matches the paths used in the previous step):

```sh
export BUILD_DIR=$HOME/path/to/llvm-project/build
mkdir build
cd build
cmake -G Ninja .. \
    -DMLIR_DIR=$BUILD_DIR/lib/cmake/mlir
ninja
```

This will build three executables: `bril2mlir`, `mlir2bril`, and `bril-opt`.

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
