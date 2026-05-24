# Thread-Escape Analysis (`examples/escape.py`)

This standalone Python pass performs an **intraprocedural escape
analysis** for the Bril concurrency extension.

* A pointer “escapes” if it is:
  * passed to `spawn`, `call`, or returned, **or**
  * stored into shared memory.
* We propagate the escape property through `ptradd` and `load`.

The script prints per-function and total thread-local allocation counts
with `--dump`, and otherwise streams the unmodified program, so it can be
dropped into any tool pipeline:

```bash
bril2json < foo.bril | python3 examples/escape.py --dump | brili …