#!/usr/bin/env python3
"""
Thread-escape analysis for Bril concurrency (analysis-only version).

It *does not* modify the program; it emits the original JSON unchanged
and writes a summary to stderr.
"""

import sys, json, argparse
from collections import deque, defaultdict


# ───────── helper predicates ─────────

def dest(instr):      return instr.get("dest")
def uses(instr):      return instr.get("args", [])
def is_alloc(instr):  return instr.get("op") == "alloc"


# ───────── escape analysis (intraprocedural) ─────────

def analyse(fn):
    """
    Return set of alloc destination names that *escape* in function `fn`.
    Conservative algorithm:
      • Any pointer passed to spawn/call/store/ret escapes.
      • Propagate escape through ptradd/load chains.
    """
    defs  = {dest(i): i for i in fn["instrs"] if dest(i)}
    esc   = set()
    work  = deque()

    def mark(v):
        if v not in esc:
            esc.add(v); work.append(v)

    # seed: obvious escapers
    for ins in fn["instrs"]:
        op = ins["op"]
        if op in ("spawn", "call", "ret"):
            for v in uses(ins):
                mark(v)
        elif op == "store":
          # args = [dst_pointer, value]
            if len(ins.get("args", [])) == 2:
                val = ins["args"][1]
                mark(val)          # value may be a pointer that escapes

    # propagate through derived pointers
    while work:
        v = work.popleft()
        d = defs.get(v)
        if not d:
            continue            # argument, already escapes
        if d["op"] == "ptradd":
            mark(d["args"][0])  # base pointer escapes
        if d["op"] == "load":
            mark(d["args"][0])  # pointer we loaded from escapes

    return esc


# ───────── driver ─────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true",
                    help="print per-function stats to stderr")
    args = ap.parse_args()

    prog = json.load(sys.stdin)
    totals_local = totals_alloc = 0

    for fn in prog["functions"]:
        escapes = analyse(fn)
        allocs  = [i for i in fn["instrs"] if is_alloc(i)]
        n_esc   = sum(1 for i in allocs if dest(i) in escapes)
        n_all   = len(allocs)
        n_loc   = n_all - n_esc
        totals_local += n_loc
        totals_alloc += n_all

        if args.dump and n_all:
            sys.stderr.write(
                f"[escape] {fn['name']}: "
                f"{n_loc}/{n_all} allocs are thread-local\n")

    if args.dump and totals_alloc:
        pct = 100 * totals_local / totals_alloc
        sys.stderr.write(
            f"[escape] TOTAL: {totals_local}/{totals_alloc} "
            f"({pct:.1f} %) thread-local\n")

    # Emit the *unchanged* program
    json.dump(prog, sys.stdout, indent=2)


if __name__ == "__main__":
    main()