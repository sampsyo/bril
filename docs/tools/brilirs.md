Fast Interpreter in Rust
========================

The `brilirs` directory contains a fast Bril interpreter written in [Rust][].
It is a drop-in replacement for the [reference interpreter](interp.md) that prioritizes speed over completeness and hacakability.
It implements [core Bril](../lang/core.md) and the [SSA][], [memory][], and [floating point][float] extensions.

Read [more about the implementation][blog], which is originally by Wil Thomason and Daniel Glus.

Install
-------
To use `brilirs` you will need to [install Rust](https://www.rust-lang.org/tools/install) and add the nightly channel with `rustup toolchain install nightly`. Use `echo $PATH` to check that `$HOME/.cargo/bin` is on your [path](https://unix.stackexchange.com/a/26059/61192).

In the `brilirs` directory, build the interpreter with:

    cargo +nightly install --path .

Run a program by piping a JSON Bril program into it:

    bril2json < myprogram.bril | brilirs


[rust]: https://www.rust-lang.org
[ssa]: ../lang/ssa.md
[memory]: ../lang/memory.md
[float]: ../lang/float.md
[blog]: https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/faster-interpreter/
