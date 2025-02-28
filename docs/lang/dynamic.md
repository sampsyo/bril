Dynamic Types
=============

This extension adds dynamic typing to Bril.
Here are some potential use cases:

* Compiling from dynamic languages (where static types are not available).
* Prototyping more complex language features that Bril's simple type system cannot yet capture.
* An "escape hatch" for situations where Bril's type system is too conservative.


Types
-----

The dynamic extension adds one new base type:

    "any"

Any instruction's result type may be `any`.
Changing an existing instruction's type from a different type to `any` does not change its run-time behavior (with one possible exception, outlined below).
It is legal to use an `any`-typed variable as an argument to any instruction;
the type checking must then occur at run time.


Well Formedness
---------------

Well-formed Bril programs using the `any` type must dynamically obey the typing constraints that would otherwise be enforced statically.
For example, this program is ill-formed:

    v: any = const 4;
    b: bool = and v v;  # Attempting to use an int as a bool.

The `main` function may not have arguments with the `any` type (because these types control how to parse command-line arguments).


Constants
---------

The `any` type may appear in `const` instructions.
The type of the resulting value is inferred using the JSON type of the `value` field.
For example, JSON Booleans become values of Bril's `bool` type, and integer values use Bril's `int`.

This leads to one possibly unexpected behavior.
The [floating point extension][float] allows constants using any JSON numerical values, including integer literals.
For example, both of these are legal and produce values of different types:

    i: int = const 4;
    f: float = const 4;

However, when using `any`, an integer literal always becomes a value of type `int`.
So converting the latter instruction to this:

    a: any = const 4;

Produces a value of type `int`, not `float`.
Use a decimal literal in the `value` field instead, such as `4.0`, to produce a `float`-typed value in an `any`-typed variable.

(This might be a mistake, and it may be a good idea to disallow `any` in `const` instructions in the same way that `main` parameters must come with a specific type.)


Interactions with the Memory Extension
--------------------------------------

The [memory extension][mem] lets you allocate "a uniformly-typed region of values."
Using the `ptr<any>` type, you can create heterogeneously typed arrays.
Here's an example:

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

[float]: ./float.md
[mem]: ./memory.md
