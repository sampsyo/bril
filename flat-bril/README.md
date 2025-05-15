# A Flattened Representation for Bril

(Final project for [CS 6120 (Advanced Compilers)](https://www.cs.cornell.edu/courses/cs6120/2025sp/https://www.cs.cornell.edu/courses/cs6120/2025sp/), Spring 2025.)

This repo contains: 
- A [flattened](https://www.cs.cornell.edu/~asampson/blog/flattening.html) representation for the [Bril IR](https://capra.cs.cornell.edu/bril/)
- An interpreter that works natively over the flattened Bril representation
- Infrastructure for converting to/from Bril's canonical JSON format to the flattened representation

For more details, see the [blog post](https://www.cs.cornell.edu/courses/cs6120/2025sp/blog/flat-bril/) for this project!

## Repo structure
- [`main.rs`](./src/main.rs): Reads in a JSON Bril file from `stdin`
- [`flatten.rs`](./src/flatten.rs): Converts a JSON Bril file to a flattened instruction format 
- [`unflatten.rs`](./src/unflatten.rs): Converts a flattened Bril instruction back to JSON
- [`memfile.rs`](./src/memfile.rs): Serializes/De-serializes a flattened Bril file to/from disk
- [`interp.rs`](./src/interp.rs): Bril interpreter which works over the flattened Bril representation
- [`types.rs`](./src/flatten.rs): Type definitions & pretty-printers
- [`json_roundtrip.rs`](.src/json_round_trip.rs): Round-trip tests for converting from JSON -> flat format -> JSON
- [`bench.py`](./bench.py), [`plot_results.py`](./plot_results.py), [`bench.sh`](./bench.sh): Miscellaneous Python/Bash scripts for running benchmarks (using [`Hyperfine`](https://github.com/sharkdp/hyperfine)) and plotting

The [`test`](./test/) subdirectory contains the [Core Bril](https://capra.cs.cornell.edu/bril/lang/core.html) benchmarks on which we tested our implementation and
compared its performance to the reference [TypeScript](https://capra.cs.cornell.edu/bril/tools/interp.html) / [Rust Brili](https://capra.cs.cornell.edu/bril/tools/brilirs.html) interpreters. 

## Command-line interface
- To install `flat-bril`, run `cargo install --path .` and make sure `$HOME/.cargo/bin` is on your path. 
- Run `flat-bril --help` to see all the supported flags. 

Here are some examples:      
- To create a flattened Bril file `(.fbril)` from an existing Bril file (eg. on `call.bril`):
```bash
$ bril2json < test/call.bril | cargo run -- --filename test/call.fbril --fbril
```
- To interpret a flattened Bril file:
```bash 
$ cargo run -- --filename test/call.fbril --interp
```
- To check that the JSON round-trip test works for a single Bril file:
```bash 
$ bril2json < test/call.bril | cargo run -- --json
```

## Building & Testing
- This repo compiles using `cargo build`. Run `cargo doc --open` to see documentation for internal functions.
- Run `turnt -e interp test/*.bril` to check that our flattened interpreter returns the same result as the reference Brili interpreter on the Core Bril benchmarks
- Run `turnt -e json test/*.bril` to run JSON round-trip tests on all the Core Bril benchmarks


***

**Other stuff in the repo (existing Bril infrastructure): **
- [`bril-rs`](./bril-rs/): existing JSON to Bril infra
- [`brilirs`](./brilirs/): existing Bril interpreter 
- [`bril2json`](./bril-rs/bril2json/): existing `bril2json` tool

