Bril: A Compiler Intermediate Representation for Learning
=========================================================

Bril (the Big Red Intermediate Language) is a compiler IR made for teaching a compilers course.
It is an extremely simple instruction-based IR that is meant to be extended.
Its canonical representation is JSON, which makes it easy to build tools from scratch to manipulate it.

This repository contains some infrastructure for Bril.
These things are written in TypeScript:

- A definition of the JSON format in `bril.ts`.
- A compiler from a very small subscript of TypeScript to Bril (`ts2bril`).
- An interpreter (`brili`).

And there is also a parser and dumper for a human-readable and -writable text format, written in Python, under `bril-txt`.


Tests
-----

There are some tests in the `test/` directory.
They use [Cram][], which lets us write the expected output for each shell command.
Install it with [pip][]:

    $ pip install --user cram

Then run the tests:

    $ cram test/*.t

Or simply:

    $ make test

[cram]: https://bitheap.org/cram/
[pip]: https://packaging.python.org/tutorials/installing-packages/
