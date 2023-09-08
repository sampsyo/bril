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

Type
----

    "<string>"
    {"<string>": <Type>}

There are two kinds of types: primitive types, whose syntax is just a string, and parameterized types, which wrap a smaller type.
The semantics chapters list the particular types that are available---for example, [core Bril](core.md) defines the basic primitive types `int` and `bool`
and the [memory extension](memory.md) defines a parameterized pointer type.

Function
--------

    {
      "name": "<string>",
      "args": [{"name": "<string>", "type": <Type>}, ...]?,
      "type": <Type>?,
      "instrs": [<Instruction>, ...]
    }

A Function object represents a (first-order) procedure consisting of a sequence of instructions.
There are four fields:

* `name`, a string.
* `args`, optionally, a list of arguments, which consist of a `name` and a `type`. Missing `args` is the same as an empty list.
* Optionally, `type`, a Type object: the function's return type, if any.
* `instrs`, a list of Label and Instruction objects.

When a function runs, it creates an activation record and transfers control to the first instruction in the sequence.

A Bril program is executable if it contains a function named `main`.
When execution starts, this function will be invoked.
The `main` function can have arguments (which implementations may supply using command-line arguments) but must not have a return `type`.

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
  (See the [*Core Language*](core.md) section and the subsequent extension sections for listings of the available opcodes.)

Depending on the opcode, the instruction might also have:

* `dest`, a string: the name of the variable where the operation's result is stored.
* `type`, a Type object: the type of the destination variable.
* `args`, a list of strings: the arguments to the operation. These are names of variables.
* `funcs`, a list of strings: any names of functions referenced by the instruction.
* `labels`, a list of strings: any label names referenced by the instruction.

There are three kinds of instructions: constants, value operations, and effect operations.

### Constant

    { "op": "const", "dest": "<string>", "type": <Type>,
      "value": <literal> }

A Constant is an instruction that produces a literal value.
Its `op` field must be the string `"const"`.
It has the `dest` and `type` fields described above, and also:

* `value`, the literal value for the constant.
  This is either a JSON number or a JSON Boolean value.
  The `type` field must match—i.e., it must be "int" or "bool", respectively.

### Value Operation

    { "op": "<string>", "dest": "<string>", "type": <Type>,
      "args": ["<string>", ...]?,
      "funcs": ["<string>", ...]?,
      "labels": ["<string>", ...]? }

A Value Operation is an instruction that takes arguments, does some computation, and produces a value.
Like a Constant, it has the `dest` and `type` fields described above, and also any of these three optional fields:

* `args`, a list of strings.
  These are variable names defined elsewhere in the same function.
* `funcs`, a list of strings.
  The names of any functions that this instruction references. For example, [core Bril](core.md)'s call instruction takes one function name.
* `labels`, a list of strings.
  The names of any labels within the current function that the instruction references. For example, [core Bril](core.md)'s jump and branch instructions have target labels.

In all three cases, these keys may be missing and the semantics are identical to mapping to an empty list.

### Effect Operation

    { "op": "<string>",
      "args": ["<string>", ...]?,
      "funcs": ["<string>", ...]?,
      "labels": ["<string>", ...]? }

An Effect Operation is like a Value Operation but it does not produce a value.
It also has the optional `args`, `funcs`, and `labels` fields.

Source Positions
----------------

Any syntax object may optionally have position fields to reflect a source position:

    { ..., "pos": {"row": <int>, "col": <int>},
           "pos_end": {"row": <int>, "col": <int>}?,
           "src": "<string>"? }

The `pos` and `pos_end` objects have two keys: `row` (the line number) and `col` (the column number within the line). The `src` object can optionally provide the absolute path to a file which is referenced to by the source position.
If `pos_end` is provided, it must be equal to or greater than `pos`.
Front-end compilers that generate Bril code may add this information to help with debugging.
The [text format parser](../tools/text.md), for example, can optionally add source positions.
However, tools can't require positions to exist, to consistently exist or not on all syntax objects in a program, or to follow any particular rules.
