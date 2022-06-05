Rust Library
============

This is a no-frills interface between Bril's JSON and your [Rust][] code. It supports the [Bril core][core] along with the [SSA][], [memory][], [floating point][float], [speculative execution][spec], and [source positions][pos] extensions.

Use
---

Include this by adding the following to your `Cargo.toml`:

```toml
[dependencies.bril-rs]
version = "0.1.0"
path = "../bril-rs"
features = ["ssa", "memory", "float", "speculate", "position"]
```

Each of the extensions to [Bril core][core] is feature gated. To ignore an extension, remove its corresponding string from the `features` list.

There are two helper functions: `load_program` will read a valid Bril program from stdin, and `output_program` will write your Bril program to stdout. Otherwise, this library can be treated like any other [serde][] JSON representation.

Tools
-----

This library supports fully compatible Rust implementations of `bril2txt` and `bril2json`. This library also implements the [import][] extension with a static linker called `brild`.

For ease of use, these tools can be installed and added to your path by running the following in `bril-rs/`:

    $ make install

Make sure that `~/.cargo/bin` is on your path. Each of these tools supports the `--help` flag which specifies some helpful flags.

Development
-----------

To maintain consistency and cleanliness, run:

```bash
cargo fmt
cargo clippy
cargo doc
make test
make features
```

[rust]: https://www.rust-lang.org
[serde]: https://github.com/serde-rs/serde
[core]: ../lang/core.md
[ssa]: ../lang/ssa.md
[memory]: ../lang/memory.md
[float]: ../lang/float.md
[spec]: ../lang/spec.md
[pos]: ../lang/syntax.md
[import]: ../lang/import.md
