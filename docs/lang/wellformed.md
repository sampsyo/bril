Well Formedness
===============

Not every syntactically complete Bril program is *well formed*.
Here is an incomplete list of rules that well-formed Bril programs must follow:

* Instructions may name variables as arguments when they are defined elsewhere in the function. Similarly, they may only refer to labels that exist within the same function, and they can only refer to functions defined somewhere in the same file.
* Dynamically speaking, during execution, instructions may refer only to variables that have already been defined earlier in execution. (This is a dynamic property, not a static property.)
* Every variable may have only a single type within a function. It is illegal to have two assignments to the same variable with different types, even if the function's logic guarantees that it is impossible to execute both instructions in a single call.
* Many operations have constraints on the types of arguments they can take; well-formed programs always provide the right type of value.

Tools do not need to handle ill-formed Bril programs.
As someone working with Bril, you never need to check for well-formedness and can do anything when fed with ill-formed code, including silently working just fine, producing ill-formed output, or crashing and burning.

To help check for well-formedness, the [reference interpreter](../tools/interp.md) has many dynamic checks and the [type inference tool](../tools/infer.md) can check types statically.
