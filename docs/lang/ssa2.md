# Static Single Assignment (SSA) Form (New!)

This language extension lets you represent Bril programs in [static single assignment (SSA)][ssa] form.
An SSA-form Bril program contains only one assignment per variable, globally—that is, variables within a function cannot be reassigned.

This extension uses [Filip Pizlo's "phi/upsilon" variant of SSA][pizlo].
In particular, it splits the traditional φ-node into a `set` instruction and a `get` instruction for communicating conditionally across control-flow edges.
(In [the "Pizlo form" description][pizlo], `set` is called "upsilon" and `get` is called "phi.")

This extension is a successor to an older, deprecated ["classic" SSA extension][ssa1], which has a more standard-looking `phi` instruction.

[ssa]: https://en.wikipedia.org/wiki/Static_single_assignment_form
[ssa1]: ./ssa.md
[pizlo]: https://gist.github.com/pizlonator/79b0aa601912ff1a0eb1cb9253f5e98d

`set` and `get` Operations
--------------------------

Semantically, the instructions in this extension adds "shadow variables" to the state of the program.
Imagine a separate environment (a map from shadow variable names to ordinary values);
the `set` instruction writes to shadow variables and `get` reads from them.
Shadow variables obey a static single use (SSU) restriction, i.e., there may be only one `get` in a function that reads from each shadow variable.

Here are the two main instructions in detail:

- `set`:
  An effect instruction that takes two arguments: a destination and a source.
  The instruction copies a value from the source, a "regular" variable, to the destination, a shadow variable.
  For example, `set x y` takes the value from the ordinary variable `y` and copies into the shadow variable `x`.
- `get`:
  A value instruction with zero arguments.
  The instruction gets the value from the shadow variable named by the destination and copies it to the regular variable of the same name.
  So, for example, `x: int = get` copies the value from the shadow variable `x` and puts it into the ordinary variable `x`.


Examples
--------

Here's a simple example that uses `set` and `get` to copy something through a shadow variable:

    one: int = const 1;
    set shadow one;
    shadow: int = get;
    print shadow;

While `set` and `get` can appear anywhere in a program, the actual purpose is to put them around control flow edges.
Put `set` at the end of one basic block and `get` at the beginning of a successor block.
Here's a small example:

      a: int = const 5;
      set c a;
      br cond .here .there;
    .here:
      b: int = const 7;
      set c b;
    .there:
      c: int = get;
      print c;

You can think of these two `set` instructions as "sending" two different values to the `get` instruction for `c`.
Because SSA allows only one assignment to every variable, `get`s are also globally unique:
that is, every function can have at most one `get` for each variable.


Undefined Values
----------------

When working with SSA form, it also turns out to be useful to explicitly propagate undefined values.
So there is one more instruction in this extension:

- `undef`:
  A value instruction with zero arguments.
  It sets the destination variable, which may be of any type, to a special *undef* value.

The only thing you can do with this undef value is copy it to other variables with `id`, `set`, and `get`.
It is an error to feed undef into any other operation.
In other words, in well-formed programs, all `undef` instructions are transitively dead.

For example, this is legal:

    x: float = undef;
    y: float = id x;

But this is not:

    x: float = undef;
    print x;


Support
-------

The [reference interpreter][interp] supports programs in SSA form because it can faithfully execute `set`, `get`, and `undef`.
This extension is also supported in [the fast Rust-based interpreter][brilirs] and [the type checker][brilck].

[interp]: ../tools/interp.md
[brilirs]: ../tools/brilirs.md
[brilck]: ../tools/brilck.md
