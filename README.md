Bril: A Compiler Intermediate Representation for Learning
=========================================================

Bril (the Big Red Intermediate Language) is a compiler IR made for teaching [CS 6120][cs6120], a grad compilers course.
It is an extremely simple instruction-based IR that is meant to be extended.
Its canonical representation is JSON, which makes it easy to build tools from scratch to manipulate it.

This repository contains the [documentation][docs], including the [language reference document][langref], and some infrastructure for Bril.
There are some quick-start instructions below for some of the main tools, but
check out the docs for more details about what's available.

[docs]: https://capra.cs.cornell.edu/bril/
[langref]: https://capra.cs.cornell.edu/bril/lang/index.html
[brilts]: https://github.com/sampsyo/bril/blob/master/bril-ts/bril.ts


Install the Tools
-----------------

### Reference Interpreter

You will want the IR interpreter, which uses [Deno][].
Just type this:

    $ deno install brili.ts

As Deno tells you, you will then need to add `$HOME/.deno/bin` to [your `$PATH`][path].
You will then have `brili`, which takes a Bril program as JSON on stdin and executes it.

[deno]: https://deno.land
[path]: https://unix.stackexchange.com/a/26059/61192

### Text Format

The parser & pretty printer for the human-editable text form of Bril are written for Python 3.
To install them, you need [Flit][], so run this:

    $ pip install --user flit

Then, go to the `bril-txt` directory and use Flit to install symlinks to the tools:

    $ flit install --symlink --user

The tools are called `bril2json` and `bril2txt`.
They also take input on stdin and produce output on stdout.

[flit]: https://flit.readthedocs.io/


Tests
-----

There are some tests in the `test/` directory.
They use [Turnt][], which lets us write the expected output for individual commands.
Install it with [pip][]:

    $ pip install --user turnt

Then run all the tests by typing `make test`.

[pip]: https://packaging.python.org/tutorials/installing-packages/
[cs6120]: https://www.cs.cornell.edu/courses/cs6120/2020fa/
[turnt]: https://github.com/cucapra/turnt
