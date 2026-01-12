Concurrency Extension
=====================

This extension brings *lightweight parallel threads* to Bril.

*  **`spawn`** launches a new thread and returns an opaque handle.
*  **`join`** blocks until a thread identified by such a handle finishes.
*  **`thread`** is a new primitive type used for those handles.

Threads share *heap memory* (the region allocated by the memory
extension) but **do not share local variables or SSA temporaries**.
Data races on the shared heap are *undefined behaviour*.

---

Primitive Type
--------------

| Type    | Meaning                                       |
|---------|-----------------------------------------------|
| thread  | Opaque handle returned by `spawn`, consumed by `join`. |

---

Instructions
------------

### `spawn`
tid: thread = spawn @func arg1 arg2 … argN

JSON form:
```jsonc
{
  "op":   "spawn",
  "dest": "tid",
  "type": "thread",
  "funcs": ["func"],
  "args":  ["arg1", "arg2", "..."]
}

* Creates a new thread that begins executing function @func.
* The caller’s current values of arg1 … argN are copied into the callee’s fresh environment; subsequent changes are not shared.
* The thread shares the global heap with every other thread in the
program.
* Returns a fresh thread handle that uniquely identifies the spawned
thread.

### `join`
join tid

JSON form:
```jsonc
{ 
  "op": "join",
  "args": ["tid"] 
}

* Blocks the current thread until the thread identified by tid
terminates.
* Joining an unknown handle or joining the same handle twice is a
run‑time error.

---

Well-Formedness Rules
----------------------
* spawn must provide exactly one function name in funcs.
* The number of args to spawn must equal the callee’s formal-parameter list.
* The destination of spawn must be declared with type thread.
* The sole argument to join must have type thread.

---

Memory-Sharing Semantics
-------------------------
*	Local variables (including SSA temporaries, set/get shadows,
and the call stack) are thread‑private.  They are copied at spawn
and never shared.
* Heap allocations created with alloc are global; every thread
may load or store through any pointer value it holds.
* If two threads access the same location and at least one access is a
store, the program has a data race, and the result is undefined.

---

Example
--------
```
@worker(x: int): void {
  print x;
  ret;
}

@main(): void {
  one: int = const 1;
  two: int = const 2;

  t1: thread = spawn @worker one;
  t2: thread = spawn @worker two;

  join t1;
  join t2;

  print 0;
  ret;
}
```

Possible output (order of first two lines is nondeterministic):
1
2
0