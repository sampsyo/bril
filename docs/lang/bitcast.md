Bit Casting
===========

Bril has an extension for bit-casting between `int`s and `float`s.
Thus, this extension depends on the [float] extension.

Operations
----------

There are two new instructions, both of them value operations:

- `float2bits`: Takes as input a single argument of type `float` and bitcasts
  the float to an `int`.
- `bits2float`: Takes as input a single argument of type `int` and bitcasts
  the integer to a `float`.

Syntax
------

The following JSON represents a bitcast from the float `<input variable>` to
an integer, storing the result in `<destination variable>`:

```json
{
    "op": "float2bits",
    "dest": "<destination variable>",
    "type": "int",
    "args": ["<input variable>"]
}
```

The following JSON represents a bitcast from the integer `<input variable>` to
a float, storing the result in `<destination variable>`:

```json
{
    "op": "bits2float",
    "dest": "<destination variable>",
    "type": "float",
    "args": ["<input variable>"]
}
```

Examples
--------

The following textual excerpt of Bril converts an integer into a
floating-point number (in this case, $0.1$):

```
as_int: int = const 4591870180066957722;
as_float: float = bits2float as_int;
print as_float;
```

[float]: ./float.md
