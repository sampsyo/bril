# Bril-rs

Bril-rs provides a straightforward representation of structurally valid Bril programs.

`bril_rs` provides two representations of Bril programs: `Program` and `AbstractProgram`. Both representations parse from JSON using `serde` and are included with helper functions for going between JSON and Rust.

`Program` is the recommended representation for most use-cases of this library as it implements the Bril core with the main extensions in a structured way(using enums). `AbstractProgram` is a less structured version of `Program` using strings. This is useful if you are working with a non-standard extension of Bril or are implementing your own Bril operations and don't want to modify this library.

See the full documentation with `cargo doc --open`.

This library is used to reimplement `bril2txt` and `bril2json` in Rust as a proof of concept. These tools are drop in replacements and can be installed with `make install`. Make sure `$HOME/.cargo/bin` is on your path. You can then use `--help` to check for the flags of each tool.
