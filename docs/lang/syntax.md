Syntax Reference
================

Bril programs are JSON objects that directly represent abstract syntax.
This chapter exhaustively describes the structure of that syntax.
All objects are JSON values of one sort or another.

Program
-------

    { "functions": [<Function>, ...] }

A Program is the top-level object.
It has one key:

* `functions`, a list of Function objects.

There should be at least one function with the name `main`.
When execution starts, this function will be invoked.

Type
----

Types are JSON values.
The semantics chapters list the particular types that are available---for example, [core Bril](core.md) defines the basic types `"int"` and `"bool"`.

Function
--------

    {
      "name": "<string>",
      "args": [{"name": "<string>", "type": <Type>}, ...],
      "type": <Type>?,
      "instrs": [<Instruction>, ...]
    }

A Function object represents a (first-order) procedure consisting of a sequence of instructions.
There are four fields:

* `name`, a string.
* `args`, a list of arguments, which consist of a `name` and a `type`.
* Optionally, `type`, a Type object: the function's return type, if any.
* `instrs`, a list of Label and Instruction objects.

When a function runs, it creates an activation record and transfers control to the first instruction in the sequence.

Label
-----

    { "label": "<string>" }

A Label marks a position in an instruction sequence as a destination for control transfers.
It only has one key:

* `label`, a string. This is the name that jump and branch instructions will use to transfer control to this position and proceed to execute the following instruction.

Instruction
-----------

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

### Constant

    { "op": "const", "dest": "<string>", "type": <Type>,
      "value": ... }

A Constant is an instruction that produces a literal value.
Its `op` field must be the string `"const"`.
It has the `dest` and `type` fields described above, and also:

* `value`, the literal value for the constant.
  This is either a JSON number or a JSON Boolean value.
  The `type` field must match—i.e., it must be "int" or "bool", respectively.

### Value Operation

    { "op": "<string>", "dest": "<string>", "type": <Type>,
      "args": ["<string>", ...] }

A Value Operation is an instruction that takes arguments, does some computation, and produces a value.
Like a Constant, it has the `dest` and `type` fields described above, and also:

* `args`, a list of strings.
  The strings are names interpreted according to the operation.
  They may refer to variables or labels.

### Effect Operation

    { "op": "<string>", "args": ["<string>", ...] }

An Effect Operation is like a Value Operation but it does not produce a value.
It also has an `args` field for its arguments.
