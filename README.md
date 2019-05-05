Bril: A Compiler Intermediate Representation for Learning
=========================================================

Bril (the Big Red Intermediate Language) is a compiler IR made for teaching a compilers course.
It is an extremely simple instruction-based IR that is meant to be extended.
Its canonical representation is JSON, which makes it easy to build tools from scratch to manipulate it.

This repository contains some TypeScript infrastructure for Bril:

- A definition of the JSON format in `bril.ts`.
- A compiler from a very small subscript of TypeScript to Bril (`ts2bril`).
- An interpreter (`brili`).
