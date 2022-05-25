# Brild

This project is a Rust implementation of a static linking tool for Bril programs called Brild. This is one way of leveraging the import Bril extension as described in the Bril documentation. In addition to implementing the import extension, this tool also supports importing bril text files which is not required the import extension specification.

Given an input Bril program, `brild` will resolve all of the imports of that program into a single, new Bril program which can then be run by a Bril interpreter.

Imports are resolved by providing a space-separated list of paths via the `-l/--libs` flag.

Install with `make install` using the Makefile in `bril/bril_rs` or `cargo install --path .` in this directory. Then use `brild --help` to get the help page for `brild` with all of the supported flags.
