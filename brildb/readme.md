# brildb

Interactive debugger for Bril.

Requires [Haskell Stack][stack].

`stack build` to build the program.

`stack exec brildb <file path to JSON Bril program>` to run the debugger.

## Commands

`run`: run the program to the next breakpoint or termination

`step`: execute one command in the program

`restart`: reset the program to the beginning of `main`

`scope`: list all variables currently defined and their values

`print <var>`: print the current value of a variable


[stack]: https://docs.haskellstack.org/en/stable/install_and_upgrade/
