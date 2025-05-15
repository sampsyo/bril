# A Flattened Representation for Bril

This subdirectory contains: 
- A [flattened](https://www.cs.cornell.edu/~asampson/blog/flattening.html) representation for the [Bril IR](https://capra.cs.cornell.edu/bril/)
- An interpreter that works natively over the flattened Bril representation (`.fbril` files)
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

## Command-line interface
- To install `flat-bril`, run `cargo install --path .` and make sure `$HOME/.cargo/bin` is on your path. 
- Run `flat-bril --help` to see all the supported flags. 

Here are some examples:      
- To create a flattened Bril file `(call.fbril)` from an existing Bril file (eg. on `call.bril`):
```bash
$ bril2json < call.bril | flat-bril --filename call.fbril --fbril
```
- To interpret a flattened Bril file:
```bash 
$ flat-bril --filename call.fbril --interp
```
- To check that the JSON round-trip test works for a single Bril file:
```bash 
$ bril2json < call.bril | flat-bril --json
```

## Building 
- This repo compiles using `cargo build`. 
- Run `cargo doc --open` to see the docs for internal functions.
- Tests & benchmarks are contained in the [original repo](https://github.com/ngernest/flat-bril).
