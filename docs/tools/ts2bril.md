# TypeScript-to-Bril Compiler

Bril comes with a compiler from a very small subset of [TypeScript][] to Bril called `ts2bril`.

It is not supposed to make it easy to port existing JavaScript code to Bril; it is a convenient way to write larger, more interesting programs without manually fiddling with Bril directly.
It also emits somewhat obviously inefficient code to keep the compiler simple; some obvious optimizations can go a long way.

[typescript]: https://www.typescriptlang.org

Install
-------

The TypeScript compiler uses [Deno][].
Type this:

    $ deno install --allow-env --allow-read ts2bril.ts

If you haven't already, you will then need to add `$HOME/.deno/bin` to [your `$PATH`][path].

[deno]: https://deno.land

Use
---

Compile a TypeScript program to Bril by giving a filename on the command line:

    $ ts2bril mycode.ts

The compiler supports both integers (from [core Bril](../lang/core.md)) and [floating point numbers](../lang/float.md).
Perhaps somewhat surprisingly, plain JavaScript numbers and the TypeScript `number` type map to `float` in Bril.
For integers, use [JavaScript big integers][bigint] whenever you need an integer, like this:

    var x: bigint = 5n;
    printInt(x);

    function printInt(x: bigint) {
        console.log(x);
    }

The `n` suffix on literals distinguishes integer literals, and the `bigint` type in TypeScript reflects them.

[bigint]: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/BigInt
