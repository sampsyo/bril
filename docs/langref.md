# Bril Language Reference

Bril programs are JSON objects that directly represent abstract syntax.

This language reference has two sections:
[*Syntax*](#syntax) describes the structure of Bril programs
and [*Operations*](#operations) lists all the built-in kinds of instructions and what they do.


## Syntax

This section describes the syntax elements that make up Bril programs.
All objects are JSON values of one sort or another.

### Program

    { "functions": [<Function>, ...] }

A Program is the top-level object.
It has one key:

* `functions`, a list of Function objects.

There should be at least one function with the name `main`.
When execution starts, this function will be invoked.

### Type

    "int"
    "bool"

There are two value types in Bril:

* `int`: 64-bit, two's complement, signed integers.
* `bool`: True or false.

In a Bril program, a Type is either of the two strings that name the type.

### Function

    { "name": "<string>", "instrs": [<Instruction>, ...] }

A Function object represents a (first-order) procedure consisting of a sequence of instructions.
There are two fields:

* `name`, a string.
* `instrs`, a list of Label and Instruction objects.

When a function runs, it creates an empty activation record and transfers control to the first instruction in the sequence.

### Label

    { "label": "<string>" }

A Label marks a position in an instruction sequence as a destination for control transfers.
It only has one key:

* `label`, a string. This is the name that jump and branch instructions will use to transfer control to this position and proceed to execute the following instruction.

### Instruction

    { "op": "<string>", ... }

An Instruction represents a unit of computational work.
Every instruction must have this field:

* `op`, a string: the *opcode* that determines what the instruction does.
  (See the [*Operations*](#operations) section, below.)

Depending on the opcode, the instruction might also have:

* `dest`, a string: the name of the variable where the operation's result is stored.
* `type`, a Type object: the type of the destination variable.
* `args`: a list of strings: the arguments to the operation.

There are three kinds of instructions: constants, value operations, and effect operations.

#### Constant

    { "op": "const", "dest": "<string>", "type": <Type>,
      "value": ... }

A Constant is an instruction that produces a literal value.
Its `op` field must be the string `"const"`.
It has the `dest` and `type` fields described above, and also:

* `value`, the literal value for the constant.
  This is either a JSON number or a JSON Boolean value.
  The `type` field must match—i.e., it must be "int" or "bool", respectively.

#### Value Operation

    { "op": "<string>", "dest": "<string>", "type": <Type>,
      "args": ["<string>", ...] }

A Value Operation is an instruction that takes arguments, does some computation, and produces a value.
Like a Constant, it has the `dest` and `type` fields described above, and also:

* `args`, a list of strings.
  The strings are names interpreted according to the operation.
  They may refer to variables or labels.

#### Effect Operation

    { "op": "<string>", "args": ["<string>", ...] }

An Effect Operation is like a Value Operation but it does not produce a value.
It also has an `args` field for its arguments.


## Operations

This section lists the opcodes for Bril instructions.

### Arithmetic

These instructions are the obvious binary integer arithmetic operations.
They all take two arguments, which must be names of variables of type `int`, and produce a result of type `int`:

* `add`: x + y.
* `mul`: x × y.
* `sub`: x - y.
* `div`: x ÷ y.

### Comparison

These instructions compare integers.
They all take two arguments of type `int` and produce a result of type `bool`:

* `eq`: Equal.
* `lt`: Less than.
* `gt`: Greater than.
* `le`: Less than or equal to.
* `ge`: Greater than or equal to.

### Logic

These are the basic Boolean logic operators.
They take arguments of type `bool` and produce a result of type `bool`:

* `not` (1 argument)
* `and` (2 arguments)
* `or` (2 arguments)

### Control

These are the control flow operations.
Unlike most other instructions, they can take *labels* as arguments instead of just variable names:

* `jmp`: Unconditional jump. One argument: the label to jump to. 
* `br`: Conditional branch. Three arguments: a variable of type `bool` and two labels. If the variable is true, transfer control to the first label; otherwise, go to the second label.
* `ret`: Function return. Stop executing the current activation record and return to the parent (or exit the program if this is the top-level main activation record). No arguments.

None of these operations produce results (i.e., they appear as Effect Operations).

### Miscellaneous

* `id`: A type-insensitive identity. Takes one argument, which is a variable of any type, and produces the same value (which must have the same type, obvi).
* `print`: Output values to the console. Takes any number of arguments of any type and does not produce a result.
* `nop`: Do nothing. Takes no arguments and produces no result.
