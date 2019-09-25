# brildb

Interactive debugger for Bril.

Requires [Haskell Stack][stack].

`stack build` to build the debugger.

`stack exec brildb <path to JSON Bril file>` to run the debugger.

## Commands

`run`: run the program to the next breakpoint or termination

`step [NUMBER]`: execute one instruction (or n instructions) in the program

`restart`: reset the program to the beginning of `main`

`scope`: list all variables currently defined and their values

`print VAR`: print the current value of a variable

`assign VAR VALUE`: set a variable to the given value

`breakpoint LOCATION [EXPR]`: set a breakpoint at the given location (label or line number) with a condition for when it triggers (default is true)

## Breakpoint Condition Syntax

Breakpoints can be conditioned on arbitrary expressions which are represented as nestable Bril instructions. For example, to break at label `foo` when `x < y < z` holds, issue the command: `breakpoint foo (and (lt x y) (lt y z))`.


[stack]: https://docs.haskellstack.org/en/stable/install_and_upgrade/
