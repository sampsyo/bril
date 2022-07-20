# rs2bril

This project is a Rust version of the `ts2bril` tool. `rs2bril` compiles a subset of Rust to Bril. See the test cases or `example.rs` for the subset of supported syntax. The goal is to support core bril operations like integer arithmetic/comparisons, function calls, control flow like if/while, and printing. It will additionally support floating point operations like those in Bril and memory operations via Rust arrays/slices.

View the interface with `cargo doc --open` or install with `make install` using the Makefile in `bril/bril_rs`. Then use `rs2bril --help` to get the help page for `rs2bril` with all of the supported flags.
