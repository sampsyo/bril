OCaml Library
=============

The [OCaml][] `bril` library, which lives in the `bril-ocaml` directory, provides an OCaml interface and parser for Bril's JSON files.

Install
-------

To build the library, you first need to [install OCaml][inst].
Then, install the dependencies with `opam install core yojson`.

To install the bril-ocaml library:

```bash
git clone https://github.com/sampsyo/bril path/to/my/bril
opam pin add -k path bril path/to/brill/bril-ocaml
opam install bril
```

That's it! You can include it in your [Dune][] files as `bril`, like any other OCaml library.

Use
---

The interface for the library can be found in `bril.mli`—good starting points are `from_string`, `from_file`, and `to_string`.
A small code example for the library lives in the `count` subdirectory.

If you wish to make changes to the bril OCaml library, simply hack on the git clone.

When you are done, simply reinstall the package with `opam reinstall bril`. Restart the build of your
local project to pick up changes made to bril-ocaml.

For Development
---------------

[ocamlformat][ocamlformat] is recommended for style consistency. The
[dune documentation on Automatic Formatting](https://dune.readthedocs.io/en/stable/formatting.html)
has information about using ocamlformat with dune.

[ocamlformat]: https://github.com/ocaml-ppx/ocamlformat
[inst]: https://ocaml.org/docs/install.html
[ocaml]: https://ocaml.org
[dune]: https://dune.build
