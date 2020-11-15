Interpreter
===========

`brili` is the reference interpreter for Bril.
It is written in [TypeScript][].
You can find `brili` in the `bril-ts` directory in the Bril repository.

The interpreter supports [core Bril](../lang/core.md) along with the [memory](../lang/memory.md), [floating point](../lang/float.md), [SSA](../lang/ssa.md), and [speculation](../lang/spec.md) extensions.

Install
-------

To set up the interpreter, you will need [Node][] and [Yarn][].
Go to the `bril-ts` directory and do this:

    $ yarn
    $ yarn build
    $ yarn link

The last thing will install symlinks to the two utility programs, but they may not be in a standard location.
To find where these tools were installed, run `yarn global bin`.
You probably want to [add this to your `$PATH`][path].

[node]: https://nodejs.org/en/
[yarn]: https://yarnpkg.com/en/
[path]: https://unix.stackexchange.com/a/26059/61192
[typescript]: https://www.typescriptlang.org

Run
---

The `brili` program takes a Bril program as a JSON file on standard input:

    $ brili < my_program.json

It emits any `print` outputs to standard output.
To provide inputs to the main function, you can write them as command-line arguments:

    $ brili 37 5 < add.json
    42

Profiling
---------

The interpreter has a rudimentary profiling mode.
Add a `-p` flag to print out a total number of dynamic instructions executed to stderr:

    $ brili -p 37 5 < add.json
    42
    total_dyn_inst: 9
