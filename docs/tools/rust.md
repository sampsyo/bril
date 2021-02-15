Rust Library
============

This is a no-frills interface between Bril's JSON and your [Rust][] code. It supports the [Bril core][core] along with the [SSA][], [memory][], [floating point][float], and [speculative execution][spec] extensions.

Use
---

Include this by adding the following to your `Cargo.toml`:

```toml
[dependencies.bril-rs]
version = "0.1.0"
path = "../bril-rs"
features = ["ssa", "memory", "float", "speculate"]
```

Each of the extensions to [Bril core][core] is feature gated. To ignore an extension, remove its corresponding string from the `features` list.

There are two helper functions, `load_program`, which will read a valid Bril program from stdin, and `output_program` which writes your Bril program to stdout. Otherwise, this library can be treated like any other [serde][] JSON representation.

Examples
---

There is currently a very trivial example that uses this interface to create a rust implementation of `bril2txt`. Run it with `cargo run --example bril2txt`.

Development
-----------

To maintain consistency and cleanliness, run:

```bash
cargo fmt
cargo clippy
```

[rust]: https://www.rust-lang.org
[serde]: https://github.com/serde-rs/serde
[core]: ../lang/core.md
[ssa]: ../lang/ssa.md
[memory]: ../lang/memory.md
[float]: ../lang/float.md
[spec]: ../lang/spec.md
