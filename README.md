Bril: A Compiler Intermediate Representation for Learning
=========================================================

Bril (the Big Red Intermediate Language) is a compiler IR made for teaching [CS 6120][cs6120], a grad compilers course.
It is an extremely simple instruction-based IR that is meant to be extended.
Its canonical representation is JSON, which makes it easy to build tools from scratch to manipulate it.

This repository contains the [language reference document][langref] and some infrastructure for Bril.
These things are written in TypeScript:

- A [definition of the JSON format][brilts] as a TypeScript declaration.
  Even if you're not using TypeScript, this file can serve as a succinct definition of the syntax.
  It has lots of comments.
- A reference interpreter (`brili`).
  It is meant to be readable but not fast.
- A compiler from a very small subset of TypeScript to Bril (`ts2bril`).
  This is useful to make it easy to write bigger Bril programs.

And there is also a parser and dumper for a human-readable and -writable text format, written in Python, under `bril-txt`.

[langref]: https://capra.cs.cornell.edu/bril/langref.html
[brilts]: https://github.com/sampsyo/bril/blob/master/bril-ts/bril.ts


Install the Tools
-----------------

### TypeScript Compiler & IR Interpreter

To install the TypeScript compiler and IR interpreter, you will need [Node][] and [Yarn][].
Go to the `bril-ts` directory and do this:

    $ yarn
    $ yarn build
    $ yarn link

The last thing will install symlinks to the two utility programs, but they may not be in a standard location.
To find where these tools were installed, run `yarn global bin`.
You probably want to [add this to your `$PATH`][path].

The tools are `brili`, an interpreter, which takes a Bril program as JSON on stdin, and `ts2bril`, which compiles a TypeScript file given on the command line to Bril.

[node]: https://nodejs.org/en/
[yarn]: https://yarnpkg.com/en/
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
[cs6120]: https://www.cs.cornell.edu/courses/cs6120/2019fa/
[turnt]: https://github.com/cucapra/turnt
