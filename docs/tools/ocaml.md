OCaml Library
=============

The [OCaml][] `bril` library, which lives in the `bril-ocaml` directory, provides an OCaml interface and parser for Bril's JSON files.

Install
-------

To build the library, you first need to [install OCaml][inst].
Then, install the dependencies with `opam install core yojson`.

Use
---

You can include the library by adding it to the `libraries` substanza of your `dune` file.
The interface for the library can be found in `bril.mli`—good starting points are `from_string`, `from_file`, and `to_string`.
A small code example for the library lives in the `count` subdirectory.

For Development
---------------

[ocamlformat][] is recommended for style consistency.

[ocamlformat]: https://github.com/ocaml-ppx/ocamlformat
[inst]: https://ocaml.org/docs/install.html
[ocaml]: https://ocaml.org
