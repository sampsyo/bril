Fast Interpreter in Rust
========================

The `brilirs` directory contains a fast Bril interpreter written in [Rust][].
It is a drop-in replacement for the [reference interpreter](interp.md) that prioritizes speed over completeness and hacakability.
`brilirs` implements [core Bril](../lang/core.md), [SSA][], [memory][], and [floating point][float] extensions.

Read [more about the implementation][blog], which is originally by Wil Thomason and Daniel Glus.

Like any other Rust project, building is easy:

    cargo build

Run a program by piping a JSON Bril program into it:

    bril2json < myprogram.bril | cargo run

[rust]: https://www.rust-lang.org
[ssa]: ../lang/ssa.md
[memory]: ../lang/memory.md
[float]: ../lang/float.md
[blog]: https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/faster-interpreter/
