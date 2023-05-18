Character
==============

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

Printing
--------

The [core `print` operation](./core.md#miscellaneous) prints `char` values.
