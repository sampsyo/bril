# The `fastbril` bytecode interpreter

This is a bytecode spec/generator/interpreter for `bril`. It aims to
be like the typescript or rust implementation, but faster.

## To build

+ binary: `make release`
  the binary will be `./build/fastbrili`
+ doc: you need to have LaTeX installed. run `make doc`
  the doc will be `./doc/brb.pdf`
  there is a prebuilt pdf in case this is difficult



### Features
We support a superset of the behavior provided by `brili`, so options like `-p`
work exactly the same. We also support the following:
 - `-b` will read in bytecode instead of the Bril json
 - `-bo <file>` will output the bytecode to `<file>`
 - `-pr` will print the program to standard out (probably more useful with the
         `-b` option)
 - `-ni` will NOT run the interpreter.
 - `-e <file>` will emit assembly to `<file>`.

the current only supported assembly is armv8. sorry.

the compiler to asm is probably the least trustworthy part of this
whole project. The general interpreter should be pretty good, but it
is always possible there are bugs. Please report bugs to
`cs897@cornell.edu`, and there's a chance that i will fix them :)
