Bit Casting
==============

Bril has an extension for bit-casting between values of the same bit width.

Operations
----------

There is one new instruction, a value operation:

- `bitcast`:
  Takes one value of bit width $N$ and produces a value of the width-$N$ output
  type with the same bit representation. It is an error to bitcast between
  types of distinct bit widths.

Syntax
---

The following JSON represents a bitcast from `<input variable>` to `<destination
type>`, storing the result in `<destination variable>`:

```json
{
    "op": "bitcast", 
    "dest": "<destination variable>", 
    "type": "<destination type>",
    "args": ["<input variable>"]
}
```

Semantics
---------

- A `bitcast` has no side effects.
- `bitcast`ing from a value of type `T` to type `T` is equivalent to the identity function.
- Let types `T` and `U` have the same bit width. If `x` is a valid value of type
  `T`, but the bit representation of `x` is not a bit representation of a valid
  value of type `U`, the result of the bitcast is undefined. Implementations may
  choose to silently allow the cast, raise an error, or perform some other
  action.

Examples
--------

The following textual excerpt of Bril uses the [float] extension to convert an
integer into a floating-point number (in this case, $0.1$):

```
as_int: int = const 4591870180066957722;
as_float: float = bitcast as_int;
print as_float;
```


[float]: ./float.md
