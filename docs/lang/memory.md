Manually Managed Memory
=======================

While core Bril only has simple scalar stack values, the memory extension adds a manually managed heap of array-like allocations.
You can create regions, like with `malloc` in C, and it is the program's responsibility to delete them, like with `free`.
Programs can manipulate pointers within these regions; a pointer indicates a particular offset within a particular allocated region.

You can read [more about the memory extension][memblog] from its creators, Drew Zagieboylo and Ryan Doenges.

[memblog]: https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/manually-managed-memory/

Types
-----

The memory extension adds a parameterized `ptr` type to Bril:

    {"ptr": <Type>}

A pointer value represents a reference to a specific offset within a uniformly-typed region of values.

Operations
----------

These are the operations that manipulate memory allocations:

* `alloc`: Create a new memory region. One argument: the number of values to allocate (an integer). The result type is a pointer; the type of the instruction decides the type of the memory region to allocate. For example, this instruction allocates a region of integers:

      {
          "op": "alloc",
          "args": ["size"],
          "dest": "myptr",
          "type": {"ptr": "int"}
      }

* `free`: Delete an allocation. One argument: a pointer produced by `alloc`. No return value.
* `store`: Write into a memory region. Two arguments: a pointer and a value. The pointer type must agree with the value type (e.g., if the second argument is an `int`, the first argument must be a `ptr<int>`). No return value.
* `load`: Read from memory. One argument: a pointer. The return type is the pointed-to type for that pointer.
* `ptradd`: Adjust the offset for a pointer, producing a new pointer to a different location in the same memory region. Two arguments: a pointer and an offset (an integer, which may be negative). The return type is the same as the original pointer type.

It is an error to access or free a region that has already been freed.
It is also an error to access (`load` or `store`) a pointer that is out of bounds, i.e., outside the range of valid indices for a given allocation.
(Doing a `ptradd` to produce an out-of-bounds pointer is not an error; subsequently accessing that pointer is.)

Printing
--------

It is not an error to use the [core][] `print` operation on pointers, but the output is not specified.
Implementations can choose to print any representation of the pointer that they deem helpful.

[core]: ./core.md
