TypeScript Library
==================

`bril-ts` is a [TypeScript][] library for interacting with Bril programs.
It is the basis for [the reference interpreter][interp] and [the included type checker][brilck], but it is also useful on its own.

The library includes:

* `bril.ts`: Type definitions for the Bril language.
  Parsing a JSON file produces a value of type `Program` from this module.
* `builder.ts`: A [builder][] class that makes it more convenient to generate Bril programs from front-end compilers.
* `types.ts`: A description of the type signatures for Bril operations, including the core language and all currently known extensions.

[TypeScript]: https://www.typescriptlang.org/
[interp]: ./interp.md
[brilck]: ./brilck.md
[builder]: https://en.wikipedia.org/wiki/Builder_pattern
