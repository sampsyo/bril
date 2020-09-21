# Static Single Assignment (SSA) Form

This language extension lets you represent Bril programs in [static single assignment (SSA)][ssa] form.
As in the standard definition, an SSA-form Bril program contains only one assignment per variable, globally—that is, variables within a function cannot be reassigned.
This extension adds ϕ-nodes to the language.

[ssa]: https://en.wikipedia.org/wiki/Static_single_assignment_form

Operations
----------

There is one new instruction:

- `phi`:
  Takes *n* labels and *n* arguments, for any *n*.
  Copies the value of the *i*th argument, where *i* is the index of the second-most-recently-executed label.
  (It is an error to use a `phi` instruction when two labels have not yet executed, or when the instruction does not contain an entry for the second-most-recently-executed label.)

Intuitively, a `phi` instruction takes its value according to the current basic block's predecessor.

Examples
--------

In the [text format](../tools/text.md), you can write `phi` instructions like this:

    x: int = phi a .here b .there;

The text format doesn't care how you interleave arguments and labels, so this is equivalent to (but more readable than) `phi a b .here .there`.
The "second-most-recent label" rule means that the labels refer to predecessor basic blocks, if you imagine blocks being "named" by their labels.

Here's a small example:

    .top:
      a: int = const 5;
      br cond .here .there;
    .here:
      b: int = const 7;
    .there:
      c: int = phi a .top b .here;
      print c;

A `phi` instruction is sensitive to the incoming CFG edge that execution took to arrive at the current block.
The `phi` instruction in this program, for example, gets its value from `a` if control came from the `.top` block and `b` if control came from the `.here` block.

The [reference interpreter](../tools/interp.md) can supports programs in SSA form because it can faithfully execute the `phi` instruction.
