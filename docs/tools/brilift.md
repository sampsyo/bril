Cranelift Compiler
==================

Brilift is a ahead-of-time or just-in-time compiler from Bril to native code using the [Cranelift][] code generator.
It supports [core Bril][core], [floating point][float], and the [memory extension][mem].

In AOT mode, Brilift emits `.o` files and also provides a simple run-time library.
By linking these together, you get a complete native executable.
In JIT mode, Brilift mimics an interpreter.

[cranelift]: https://github.com/bytecodealliance/wasmtime/tree/main/cranelift
[core]: ../lang/core.md
[float]: ../lang/float.md
[mem]: ../lang/memory.md

Build
-----

Brilift is a Rust project using the [bril-rs][] library.
You can build it using [Cargo][]:


    $ cd brilift
    $ cargo run -- --help
    $ cargo install --path .  # If you want the executable on your $PATH.

[bril-rs]: rust.md
[cargo]: https://doc.rust-lang.org/cargo/

Ahead-of-Time Compilation
-------------------------

Provide the `brilift` executable with a Bril JSON program:

    $ bril2json < something.bril | brilift

By default, Brilift produces a file `bril.o`.
(You can pick your own output filename with `-o something.o`; see the full list of options below.)

A complete executable will also need our runtime library, which is in `rt.c`.
There is a convenient Makefile rule to produce `rt.o`:

    $ make rt.o

Then, you will want to link `rt.o` and `bril.o` to produce an executable:

    $ cc bril.o rt.o -o myprog

If your Bril `@main` function takes arguments, those are now command-line arguments to the `myprog` executable.

Just-in-Time Compilation
------------------------

Use the `-j` flag to compile and run the program immediately:

    $ bril2json < something.bril | brilift -j

Pass any arguments to the Bril `@main` function as command-line arguments to Brilift.
For example, if you have a function `@main(foo: int, bar: bool)`, you can type `brilift -j 42 true`.

Options
-------

Type `brilift --help` to see the full list of options:

* `-j`: JIT-compile the code and run it immediately, instead of AOT-compiling an object file (the default).
* `-O [none|speed|speed_and_size]`: An [optimization level][opt_level], according to Cranelift. The default is `none`.
* `-v`: Enable lots of logging from the Cranelift library.
* `-d`: Dump the Cranelift IR text for debugging.

These options are only relevant in AOT mode:

* `-o <FILE>`: Place the output object file in `<FILE>` instead of `bril.o` (the default).
* `-t <TARGET>`: Specify the target triple, as interpreted by Cranelift. These triples resemble the [target triples][triple] that LLVM also understands, for example. For instance, `x86_64-unknown-darwin-macho` is the triple for macOS on Intel processors.

[opt_level]: https://docs.rs/cranelift-codegen/0.95.1/cranelift_codegen/settings/struct.Flags.html#method.opt_level
[triple]: https://clang.llvm.org/docs/CrossCompilation.html#target-triple
