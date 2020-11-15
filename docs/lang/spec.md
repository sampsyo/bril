# Speculative Execution

This extension lets Bril programs use a form of explicit [speculative execution][spec] with rollback.

In general, *speculation* is when programs perform work that might not actually be necessary or even correct, under the assumption that it is *likely* to be right and useful.
If this assumption turns out to be wrong, speculation typically needs some *rollback* mechanism to undo incorrect side effects and recover to a correct state.

In this Bril extension, programs can explicitly enter a *speculative mode*, where variable assignments are temporary.
Then, they can either abort or commit those assignments, discarding them or making them permanent.


Operations
----------

* `speculate`: Enter a speculative execution context. No arguments.
* `commit`: End the current speculative context, committing the current speculative state as the "real" state. No arguments.
* `guard`: Check a condition and possibly abort the current speculative context. One argument, the Boolean condition, and one label, to which control is transferred on abort. If the condition is true, this is a no-op. If the condition is false, speculation aborts: the program state rolls back to the state at the corresponding `speculate` instruction, execution jumps to the specified label.

Speculation can be nested, in which case aborting or committing a child context returns execution to the parent context.
Aborting speculation rolls back normal variable assignments, but it does not affect the [memory extension][mem]'s heap—any changes there remain.
It is an error to commit or abort outside of speculation.
It is not an error to perform side effects like `print` during speculation, but it is probably a bad idea.


Examples
--------

Committing a speculative update makes it behave like normal:

    v: int = const 4;
    speculate;
    v: int = const 2;
    commit;
    print v;

So this example prints `2`.
However, when a guard fails, it rolls back any modifications that happened since the last `speculate` instruction:

      b: bool = const false;

      v: int = const 4;
      speculate;
      v: int = const 2;
      guard b .failed;
      commit;

    .failed:
      print v;

The guard here fails because `b` is false, then `v` gets restored to its pre-speculation value, and then control transfers to the `.failed` label.
So this example prints `4`.
You can think of the code at `.failed` as the "recovery routine" that handles exceptional conditions.


Interpreter
-----------

The [reference interpreter][interp] supports speculative execution.
However, it does not support function calls during speculation, so you will get an error if you try to use a `call` or `ret` instruction while speculating.

[spec]: https://en.wikipedia.org/wiki/Speculative_execution
[mem]: memory.md
[interp]: ../tools/interp.md
