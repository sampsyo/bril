OCaml library
=============

The OCaml `bril` library, which lives in the `bril-ocaml` directory, provides an OCaml interface and parser for Bril's JSON files.

Install
-------

To build the library, you first need to install [OCaml][].
Then, install dependencies with `opam install core yojson`.

[ocaml]: https://ocaml.org/docs/install.html

Usage
-----

The current interface contains a function `val parse : string -> Bril.t`, which parses a JSON string into an OCaml value of type `Bril.t` representing a Bril program. You can include the library by adding it to the `libraries` substanza of your `dune` file.

A small code example for the library lives in the `count` subdirectory.

For Development
---------------

[ocamlformat][] is recommended for style consistency.

[ocamlformat]: https://github.com/ocaml-ppx/ocamlformat
