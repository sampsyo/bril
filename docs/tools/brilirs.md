Fast Interpreter in Rust
========================

The `brilirs` directory contains a fast Bril interpreter written in [Rust][].
It is a drop-in replacement for the [reference interpreter](interp.md) that prioritizes speed over completeness and hackability.
It implements [core Bril](../lang/core.md) along with the [SSA][], [memory][], [char][], and [floating point][float] extensions.

Read [more about the implementation][blog], which is originally by Wil Thomason and Daniel Glus.

Install
-------

To use `brilirs` you will need to [install Rust](https://www.rust-lang.org/tools/install). Use `echo $PATH` to check that `$HOME/.cargo/bin` is on your [path](https://unix.stackexchange.com/a/26059/61192).

In the `brilirs` directory, install the interpreter with:

    $ cargo install --path .

During installation, `brilirs` will attempt to create a tab completions file for current shell. If this of interest, follow the instructions provided as a warning to finish enabling this.

Run a program by piping a JSON Bril program into it:

    $ bril2json < myprogram.bril | brilirs

or

    $ brilirs --text --file myprogram.bril

Similar to [brilck](brilck.md), `brilirs` can be used to typecheck and validate your Bril JSON program by passing the `--check` flag (similar to `cargo --check`).

To see all of the supported flags, run:

    $ brilirs --help

[rust]: https://www.rust-lang.org
[ssa]: ../lang/ssa.md
[memory]: ../lang/memory.md
[float]: ../lang/float.md
[char]: ../lang/char.md
[blog]: https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/faster-interpreter/
