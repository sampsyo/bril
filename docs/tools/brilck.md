# Type Checker

Bril comes with a simple type checker to catch errors statically.
It checks the types of instructions in the [core language](../lang/core.md) and the [floating point](../lang/float.md), [SSA](../lang/ssa.md), [memory](../lang/memory.md), and [speculation](../lang/spec.md) extensions.
It also checks calls and return values and the labels used in control flow.

Install
-------

The `brilck` tool comes with the same [TypeScript][] package as the [reference interpreter](interp.md).
Follow [those instructions](interp.md#install) to install it.

[typescript]: https://www.typescriptlang.org

Check
-----

Just pipe a Bril program into `brilck`:

    bril2json < benchmarks/fizz-buzz.bril | brilck

It will print any problems it finds to standard error.
(If it doesn't find any problems, it doesn't print anything at all.)

You can optionally provide a filename as a (sole) command-line argument.
This filename will appear in any error messages for easier parsing when many files are involved.
