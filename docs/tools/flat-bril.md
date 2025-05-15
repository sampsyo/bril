# Flattened Representation 

The `flat-bril` directory contains: 
- A [flattened][adrian-blog] representation for Bril
- An interpreter that works natively over the flattened Bril representation (`.fbril` files)
  - Note: the interpreter currently only supports [core Bril](./lang/core.md).
- Infrastructure for converting to/from Bril's canonical JSON format to the flattened representation

Read [more about the implementation][blog], which is originally by Ernest Ng, Sam Breckenridge and Katherine Wu.

**Caution**: The `flat-bril` interpreter is *not* meant to be a drop-in replacement for the reference Bril interpreters ([`brili`](./interp.md) and [`brilirs`](./brilirs.md)), as `flat-bril` only supports [core Bril](./lang/core.md). For most situations, we recommend using the reference Bril interpreter. 

## Install
To use `flat-bril` you will need to [install Rust][install-rust]. Use `echo $PATH` to check that `$HOME/.cargo/bin` is on your [path][path].

In the `flat-bril` directory, install the tool with `cargo install --path .`.

## Usage
Run `flat-bril --help` to see all the supported command-line flags. 

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

[adrian-blog]: https://www.cs.cornell.edu/~asampson/blog/flattening.html
[blog]: https://www.cs.cornell.edu/courses/cs6120/2025sp/blog/flat-bril/
[install-rust]: https://www.rust-lang.org/tools/install
[path]: https://unix.stackexchange.com/a/26059/61192

