Rust Library
============

This is a no-frills interface between bril's json and your rust code. It supports the bril core syntax along with SSA, Memory, and Floating Point extensions.

Use
---

Include this by adding the following to your Cargo.toml

```toml
[dependencies.bril-rs]
version = "0.1.0"
path = "../bril-rs"
```

There is one helper function ```load_program``` which will read a valid bril program from stdin and return it as a Rust struct. Otherwise, this library can be treated like any other [serde](https://github.com/serde-rs/serde) json representation.

Development
-----------

To maintain consistency and cleanliness, run:

```bash
cargo fmt
cargo clippy
```
