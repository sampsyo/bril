Dynamic
==============

This extension enhances the Bril type system to better capture the strong,
(mostly-)dynamic typing of Bril programs.

There are two types of intended use of this extension:
- Giving an escape hatch to valid Bril programs whose dynamic typing can be
  rejected by a conservative type analysis
- Frontend users who want to compile down higher-level programming abstractions
  into Bril(structs, enums, dynamic dispatch, (limited)higher-order functions)

Types
-----

The dynamic extension adds one new base type:

    "any"

Any Bril value can be typed statically with `any`. It does not change the runtime behavior
of the value. Values given the type `any` can still be used as their runtime
type with typed operations.

Well-formedness
---------------

Well-formed Bril programs using this extension only use values with the `any`
type in correspondence to their actual runtime typing.

For example, the following program is ill-formed because of the second line.

```
v: any = const 4;
b: bool = and v v; # Attempting to use an int as a bool
```

Interactions with float extension
----------------------------------

The floating point extension allows for an implicit cast of integer literals to floats
during a `const` operation.

```
i: int = const 4;
f: float = const 4;
print i f; # `4 4.00000000000000000`
```

Ascribing the type of `any` to an integer literal creates an integer value.

```
i: int = const 4;
a: any = const 4;
f: float = const 4;
print i a f; # `4 4 4.00000000000000000`
```

Interactions with memory extension
----------------------------------

The memory extension expects `a uniformly-typed region of values` which is
enforced at runtime. The dynamic extension modestly relaxes this restriction by
allowing arrays with a base type of `any` like `ptr<any>` or `ptr<ptr<any>>`.
This enables heterogeneous arrays at runtime.

```
@main {
  v: int = const 2;
  o1: int = const 1;
  bp: ptr<any> = alloc v;
  bp2: ptr<any> = ptradd bp o1;
  b: bool = const true;
  i: int = const 0;
  store bp b;
  store bp2 i;
  b: bool = load bp;
  i: int = load bp2;
  print b i; # prints true 0
  free bp;
}
```
