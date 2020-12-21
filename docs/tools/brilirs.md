Fast Interpreter in Rust
========================

The `brilirs` directory contains a fast Bril interpreter written in [Rust][].
It is a drop-in replacement for the [reference interpreter](interp.md) that prioritizes speed over completeness and hacakability.
It only [core Bril](../lang/core.md), except for functions, and the [floating point](../lang/float.md) extension.

Read [more about the implementation][blog], which is originally by Wil Thomason and Daniel Glus.

Like any other Rust project, building is easy:

    cargo build

Run a program by piping a JSON Bril program into it:

    cargo run < myprogram.json

[rust]: https://www.rust-lang.org
[blog]: https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/faster-interpreter/
