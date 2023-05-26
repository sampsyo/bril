Character
=========

Types
-----

The character extension adds one new base type:

    "char"

Characters are [UTF-16][] code units.


[UTF-16]: https://en.wikipedia.org/wiki/UTF-16

Operations
----------

Comparison operators, which take two `char` values and produce a `bool`:

- `ceq`
- `clt`
- `cle`
- `cgt`
- `cge`

Conversion operators:

- `char2int`: One argument: a variable of type `char`. Returns an integer between 0 and 65535 representing the UTF-16 code unit of the given value.
- `int2char`: One argument: a variable of type `int`. Returns the corresponding UTF-16 code unit. Throws if the value is not between 0 and 65535, inclusive.

Printing
--------

The [core `print` operation](./core.md#miscellaneous) prints `char` values.
