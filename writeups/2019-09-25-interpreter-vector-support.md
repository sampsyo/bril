+++
title = "Vector Instruction Support in the Bril Interpreter"
extra.bio = """
  Philip Bedoukian is a 3rd year PhD student in ECE. His research focuses on high-performance reconfigurable computating architectures.
"""
+++

## Problem Statement

The Bril interpreter does not take advantage of the vector instructions present on modern CPUs. This presents two problems: the first is that a Bril backend cannot generate vector instructions because they are not present in the intermediate language. The second issue is that  interpreting vector instructions is slow because it requires a loop over each vector element. We propose to support vector instructions in the interpreter and accelerate the interpretation of them using intrinsic vector instructions.

## Vector Instructions

Single-Instruction Multiple-Data(SIMD) allows computer hardware to obtain signifigant speedups over the conventional execution paradigm Multiple-Instruction Multiple-Data (MIMD). SIMD simply executes fewer instructions than MIMD to accomplish the same amount of work. In SIMD, multiple arithmetic operations are grouped under one instruction. Generally, this aritmetic is executed over multiple execution units at the same instant (spatial SIMD), although some implementations allow the same instruction to use the same functional unit over multiple cycles (temporal SIMD). Various hardware architectures allow the programmer to support SIMD by exposing vector instructions in their ISA. Typically, there is a vector register file which holds multiple elements per register and can participate in vector arithmetic.

## Vector Support in the Bril Interpreter

### Memory Support

Vector instructions are only useful when operating on large chunks of data. CPU registers generally hold no more than 32 values, so data memory is needed in the Bril interpreter to support interesting vector programs. The interpreter memory is implemented like a stack. The program is allowed to access any location in the fixed size memory. Stack frames were not implemented because the interpreter only supports a single function.

Memory access requires both loads and store operations. To emulate arrays we can loop over multiple addresses and perform and load and/or store at each location. Load and store instructions in Bril are implemented similarly to their assembly counterparts. A load takes in a memory address (an `int` register in bril) and writes to a destination register. A store is an effect operation and does not write a register. It takes two registers: a register containing a value and a register containing an address. The bril syntax is given below.

```C
// load value from address 2
addr: int = const 2;
data: int = lw addr;

// store value to address 2
sw data addr;
```

### Interpreted Vector Instructions

New vector instructions were written for the bril interpreter in Typescript. We specifically implement fixed-sized vector instructions of length four (akin to native Intel `__m128` SSE instructions). Typescript and Javascript (Typescript always compiles to Javascript) do not have support for vector instrinsics in the current standard. Thus, we implement the bril vector instructions as a loop over four values. Additionally, we add vector registers in Bril which must be used in the vector instructions. We target a vector-vector add (vvadd) program, so we include interpreter support for `vadd`, `vload`, and `vstore` instructions. The `vload` and `vstore` instructions communicate data between vector registers and the interpreter stack. The `vadd` instructions adds two vector registers and writes to a destination vector register. An example vvadd program is shown below.

```C
// locations of memory (arrays of 4 elements)
a: int = const 0;
b: int = const 4;
c: int = const 8;

// load into src vector registers
va: vector = vload a;
vb: vector = vload b;

// do the vector add
vc: vector = vadd va vb;

// store from vector register to memory
vstore vc c;
```

### Instrinsic Vector Support

It's awkward that the interpreter supports execute of a vector instruction, but doesn't actually use a native vector instruction to execute it. We expect that the performance of the interpreted vector instructions will be poor. To explore this hypothesis we create a version of an interpreted vector add instruction using three methods: 1) typescript, 2) serial c++, 3) vectorized c++. We run each test for 10,000 iterations and average the execution time over five runs. We assumed 10,000 iterations was enough time for the Typescript/Javascript JIT to warmup.

|     Method    | Time per iteration (ns) |
| ------------- | ------------- |
| Typescript  | 317  |
| Serial C++  | 16  |
| Vector C++  | 9  |

There is benefit to using native vector instruction from C (2x) speedup over serial C version. The other insteresting point is that the C version is an order of magnitude (20x) faster than the Typescript implemtation. This is expected, but is interesting most of the speedup will come from just doing the operation in C.

### C++ binding for Typescript

In order to realize this performance, we need to utilize the C++ implementations in the Bril interpreter. We use [nbind](https://github.com/charto/nbind) to allow Typescript to execute binaries generated from C++. These sorts of calls are will add potentially significant overhead to the execution. We quantify this overhead to see if it is practical in an intepreter. Note that each time we run a single vector instruction we must make a call to the binding. We run a vector add program with various iterations. Each iteration does two vector loads, a vector add, and a vector store along with instructions to facilitate the iteration. We run five configurations (128, 1024, 2048, 4196, 8192) and average the execution time over five runs. We compare the execution time with and without calls to the binding (literally just comment the line out). Note that the calls include arguments to the C++ code. On average there is a 10% overhead in the program due to the binding call. This overhead is expected to be offset from the substantial speedups offered by the C++ implementation.

## Evaluation

### Correctness

We write multiple test programs to verify that the memory and vector instructions functioned as expected. [Turnt](https://github.com/cucapra/turnt) is used to test the expected output of the program from `print` instructions. The vector programs could not be verified with Turnt, however, because it needs to be executed in the interpreter directory to find the location of the C++ binaries. These were verified by manually inspecting the output. We test a simple store and then load, multiple store and then multiple loads, and vvadd with both the Typescript implementation and the C++ vector implementation.

### Performance

We evaluate the effectiveness of using instrinic vector functions in the Bril interpreter. We run a multi-iteration vvadd program where each iteration does a single vector add of four elements. The program is shown below. 

```C
// initialize number of iterations
size: int = const 8192;
vecSize: int = const 4;

// initialize data locations
a: int = const 0;
b: int = add a size;
c: int = add b size;

// loop
i: int = const 0;
vvadd_loop:

// get base addresses to add
ai: int = add a i;
bi: int = add b i;
ci: int = add c i;

// do the vvadd
va: vector = vload ai;
vb: vector = vload bi;
vc: vector = vadd va vb;
vstore vc ci;

// iterations (increment by vector length)
i: int = add i vecSize;
done: bool = ge i size;
br done vvadd_done vvadd_loop;

vvadd_done:
```

Notice that is does not contain loads, stores, or prints and just reads from the unitialized stack. This is neccessary because the interpreter doesn't currently have an instruction to act as a timer (although this could be added in the future) and we do not to count setup and finish time of the program. We time the execution of the interpreter using Typescript's `console.time` and `console.timeEnd` functions. We take care not to time the file I/O part of the interpreter as this would dominate the runtime.

We run this with various iteration amounts (128, 1024, 2048, 4098, and 8196). We average the runtime of five executions fo the same program. Our baseline is the Typescript implementation of the vector instructions. The following figure shows the speedup of various implemenations relevant to the baseline.

PICTURE HERE

The C++ implemetations outperform the Typescript impletation at a smaller number of iterations. However, the performance becomes equals for higher iterations. This is potentially due to the Javascript JIT warming up on later iterations and matching the C++ generated code. As shown before, C++ is still expected to get much better performance up to this point though. Therefore, the JIT hypothesis does not explain the full trend.

The C++ implementations have very similar execution times even though a 2x performance gap was expected. 

Both of these results are likely due to the overhead in Typescript. The 4th series in the graph shows the speedup if the C++ calls are removed altoghter. This results ins slight speedup but not a substantial ones. We can conclude that the Typescript runtime is much higher relative to the actually vector computation. Performing optimizations on the vector computation part therefore would not provide significant speedup, which is what we are seeing for the most part.

## Conclusion

We were able to correctly implement vector instructions in the Bril interpreter. However, we were not able to obtain execution speedup of these instructions due to slowness of Typescript. If one wanted to get speedups in the interpreter it would need to be written fully in C++ (or another fast language) rather than making fine-grained calls to C++.




