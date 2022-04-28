Core Language
=============

This section describes the *core* Bril instructions.
Any self-respecting Bril tool must support all of these operations; other extensions are more optional.

Types
-----

Core Bril defines two primitive types:

* `int`: 64-bit, two's complement, signed integers.
* `bool`: True or false.

Arithmetic
----------

These instructions are the obvious binary integer arithmetic operations.
They all take two arguments, which must be names of variables of type `int`, and produce a result of type `int`:

* `add`: x + y.
* `mul`: x × y.
* `sub`: x - y.
* `div`: x ÷ y.

In each case, overflow follows two's complement rules.
It is an error to `div` by zero.

Comparison
----------

These instructions compare integers.
They all take two arguments of type `int` and produce a result of type `bool`:

* `eq`: Equal.
* `lt`: Less than.
* `gt`: Greater than.
* `le`: Less than or equal to.
* `ge`: Greater than or equal to.

Logic
-----

These are the basic Boolean logic operators.
They take arguments of type `bool` and produce a result of type `bool`:

* `not` (1 argument)
* `and` (2 arguments)
* `or` (2 arguments)

Control
-------

These are the control flow operations.
Unlike the value operations above, they take labels and functions in addition to normal arguments.

* `jmp`: Unconditional jump. One label: the label to jump to.
* `br`: Conditional branch. One argument: a variable of type `bool`. Two labels: a true label and a false label. Transfer control to one of the two labels depending on the value of the variable.
* `call`: Function invocation. Takes the name of the function to call and, as its arguments, the function parameters. The `call` instruction can be a Value Operation or an Effect Operation, depending on whether the function returns a value.
* `ret`: Function return. Stop executing the current activation record and return to the parent (or exit the program if this is the top-level main activation record). It has one optional argument: the return value for the function.

Only `call` may (optionally) produce a result; the rest appear only as Effect Operations.

Miscellaneous
-------------

* `id`: A type-insensitive identity. Takes one argument, which is a variable of any type, and produces the same value (which must have the same type, obvi).
* `print`: Output values to the console (with a newline). Takes any number of arguments of any type and does not produce a result.
* `nop`: Do nothing. Takes no arguments and produces no result.
