Floating Point
==============

Bril has an extension for computing on floating-point numbers.

You can read [more about the extension][fpblog], which is originally by [Dietrich Geisler][dietrich] and originally included two FP precision levels.

[dietrich]: https://www.cs.cornell.edu/~dgeisler/
[fpblog]: https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/floats-static-arrays/

Types
-----

The floating point extension adds one new base type:

    "float"

Floating point numbers are 64-bit, double-precision [IEEE 754][] values.
(There is no single-precision type.)

Floating point numbers are outputted with 17 decimal points of precision. Positive and negative zeros compare as equal but are distinct values when printed. Nan and Infinity are valid runtime values that are represented as `-Infinity`/`Infinity`/`NaN` respectively.

[IEEE 754]: https://en.wikipedia.org/wiki/IEEE_754

Operations
----------

There are the standard arithmetic operations, which take two `float` values and produce a new `float` value:

- `fadd`
- `fmul`
- `fsub`
- `fdiv`

It is not an error to `fdiv` by zero; as in IEEE 754, the result is infinity.

There are also comparison operators, which take two `float` values and produce a `bool`:

- `feq`
- `flt`
- `fle`
- `fgt`
- `fge`
