Manually Managed Memory
=======================

Types
-----

The memory extension adds one new form of type to Bril, whose syntax is an object:

    {"ptr": <Type>}

A pointer value represents a reference to a specific offset within a uniformly-typed region of values.

Operations
----------

These are the operations for using heap memory, which stores data that persists between
function lifetimes. These operations require the program to manually allocate heap memory for use
and to free it when they are done.

* `alloc arg1`: Allocate. Allocates `arg1` memory cells on the heap and produces a pointer to the first cell. `arg1` must be an integer.
* `free arg1`: Free (de-allocate). Releases all of the memory cells pointed to by `arg1`; those cells may no longer be read or written to. `arg1` must be a pointer which corresponds to a pointer produced by `alloc`, which has not already been freed.
* `store arg1 arg2`: Store. This writes the data, `arg2`, into the memory cell pointed to by `arg1`. `arg1` must be a valid pointer which is meant to store data of the same type as `arg2` (e.g. if `arg2` is an `int`, `arg1` must be a `ptr<int>`).
* `load arg1`: Load. This reads the data from the memory cell pointed to by `arg1` and produces that value. `arg1` must be a valid pointer.
* `ptradd arg1 arg2`: Pointer Addition. This takes the pointer `arg1` and produces a new pointer that refers to the data `arg2` memory cells further into the allocation. `arg1` must be a pointer and `arg2` must be an integer (it may be negative).
