Rust Library
============

This is a no-frills interface between Bril's JSON and your [Rust][] code. It supports the [Bril core][core] along with the [SSA][], [memory][], and [floating point][float] extensions.

Use
---

Include this by adding the following to your `Cargo.toml`:

```toml
[dependencies.bril-rs]
version = "0.1.0"
path = "../bril-rs"
```

There is one helper function, `load_program`, which will read a valid Bril program from stdin and return it as a Rust struct. Otherwise, this library can be treated like any other [serde][] JSON representation.

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
