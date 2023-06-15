Character
=========

Types
-----

The character extension adds one new base type:

    "char"

Characters are a singular [Unicode character][].


[Unicode character]: https://www.unicode.org/glossary/#character

Operations
----------

Comparison operators, which take two `char` values and produce a `bool`:

- `ceq`
- `clt`
- `cle`
- `cgt`
- `cge`

Conversion operators:

- `char2int`: One argument: a variable of type `char`. Returns an integer representing the Unicode code point of the given value.
- `int2char`: One argument: a variable of type `int`. Returns the corresponding Unicode character. Throws if the value does not correspond to a valid Unicode code point.

Printing
--------

The [core `print` operation](./core.md#miscellaneous) prints `char` values.
