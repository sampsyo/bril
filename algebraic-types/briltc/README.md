
Bril type-checker that supports algebraic data types. Build and
install the `briltc` tool with `make`.

Use like this: `cat program.bril | bril2json | briltc`. If this
terminates normally, then the input program type-checks.

Note that the raised exceptions are not necessarily useful for fixing
the typing errors in your program, although this tool could be
improved in the future to make the error messages for useful.
