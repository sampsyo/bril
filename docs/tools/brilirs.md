Fast Interpreter in Rust
========================

The `brilirs` directory contains a fast Bril interpreter written in [Rust][].
It is a drop-in replacement for the [reference interpreter](interp.md) that prioritizes speed over completeness and hackability.
It implements [core Bril](../lang/core.md) and the [SSA][], [memory][], and [floating point][float] extensions.

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

Similar to [type-infer](infer.md), `brilirs` can be used to typecheck and validate your Bril JSON program by passing the `--check` flag (similar to `cargo --check`).


        brilirs 0.1.0
        Wil Thomason <wil.thomason@gmail.com>

        USAGE:
            brilirs [OPTIONS] [ARGS]...

        ARGS:
            <ARGS>...    Arguments for the main function

        OPTIONS:
            -c, --check          Flag to only typecheck/validate the bril program
            -f, --file <FILE>    The bril file to run. stdin is assumed if file is not provided
            -h, --help           Print help information
            -p, --profile        Flag to output the total number of dynamic instructions
            -t, --text           Flag for when the bril program is in text form
            -V, --version        Print version information


[rust]: https://www.rust-lang.org
[ssa]: ../lang/ssa.md
[memory]: ../lang/memory.md
[float]: ../lang/float.md
[blog]: https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/faster-interpreter/
