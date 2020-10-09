Bril: A Compiler Intermediate Representation for Learning
=========================================================

Bril, the Big Red Intermediate Language, is a programming language for learning about compilers.
It's the intermediate representation we use in [CS 6120][cs6120], a PhD-level compilers course.
Bril's design tenets include:

- Bril is an instruction-oriented language, like most good IRs.
- The core is minimal and ruthlessly regular. Extensions make it interesting.
- The tooling is language agnostic. Bril programs are just [JSON][].
- Bril is typed.

See the [language reference](lang) for the complete language specification and the [tool documentation](tools) for details on the "batteries included" [monorepo][gh].

[cs6120]: https://www.cs.cornell.edu/courses/cs6120/2020fa/
[json]: https://www.json.org
[gh]: https://github.com/sampsyo/bril
