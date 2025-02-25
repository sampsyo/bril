import json
import sys
from collections import defaultdict

from cfg import block_map, successors, add_terminators, add_entry, reassemble
from form_blocks import form_blocks
from dom import get_dom, dom_fronts, dom_tree


def def_blocks(blocks):
    """Get a map from variable names to defining blocks."""
    out = defaultdict(set)
    for name, block in blocks.items():
        for instr in block:
            if "dest" in instr:
                out[instr["dest"]].add(name)
    return dict(out)


def get_gets(blocks, df, defs):
    """Find where to insert `get`s in the blocks.

    Produce a map from block names to variable names that need `get`s
    in those blocks. (We will need to generate names and actually insert
    instructions later.)
    """
    gets = {b: set() for b in blocks}
    for v, v_defs in defs.items():
        v_defs_list = list(v_defs)
        for d in v_defs_list:
            for block in df[d]:
                # Add a `get`...
                if v not in gets[block]:  # ...unless we already did.
                    gets[block].add(v)
                    if block not in v_defs_list:
                        v_defs_list.append(block)
    return gets


def ssa_rename(blocks, gets, succ, domtree, args):
    stack = defaultdict(list, {v: [v] for v in args})
    get_dests = {b: {p: "" for p in gets[b]} for b in blocks}
    sets = {b: [] for b in blocks}
    inits = {}
    counters = defaultdict(int)

    def _push_fresh(var):
        fresh = "{}.{}".format(var, counters[var])
        counters[var] += 1
        stack[var].insert(0, fresh)
        return fresh

    def _peek(var):
        if stack[var]:
            return stack[var][0]
        else:
            init_name = f"{var}.init"
            inits[var] = init_name
            return init_name

    def _rename(block):
        # Save stacks.
        old_stack = {k: list(v) for k, v in stack.items()}

        # Rename `get` destinations.
        for p in gets[block]:
            get_dests[block][p] = _push_fresh(p)

        # Locally "version" the writes within each block.
        for instr in blocks[block]:
            # Rename arguments in normal instructions.
            if "args" in instr:
                new_args = [_peek(arg) for arg in instr["args"]]
                instr["args"] = new_args

            # Rename destinations.
            if "dest" in instr:
                instr["dest"] = _push_fresh(instr["dest"])

        # Add `set` instructions to send values to each successor.
        for s in succ[block]:
            for p in gets[s]:
                sets[block].append((s, p, _peek(p)))

        # Recursive calls.
        for b in sorted(domtree[block]):
            _rename(b)

        # Restore stacks.
        stack.clear()
        stack.update(old_stack)

    entry = list(blocks.keys())[0]
    _rename(entry)

    return sets, get_dests, inits


def insert_sets_and_gets(blocks, sets, get_dests, types):
    for block, instrs in blocks.items():
        # Add `set`s to the bottom of the block.
        for succ, old_var, val in sorted(sets[block]):
            set_inst = {
                "op": "set",
                "args": [get_dests[succ][old_var], val],
            }
            instrs.insert(-1, set_inst)  # Before the terminator.

        # Add `get`s to the top of the block.
        for old_var, new_var in sorted(get_dests[block].items()):
            get_inst = {
                "op": "get",
                "dest": new_var,
                "type": types[old_var],
            }
            instrs.insert(0, get_inst)


def insert_inits(entry, inits, types):
    for old_var, init_var in sorted(inits.items()):
        undef = {
            "op": "undef",
            "type": types[old_var],
            "dest": init_var,
        }
        entry.insert(0, undef)


def get_types(func):
    # Silly way to get the type of variables. (According to the Bril
    # spec, well-formed programs must use only a single type for every
    # variable within a given function.)
    types = {arg["name"]: arg["type"] for arg in func.get("args", [])}
    for instr in func["instrs"]:
        if "dest" in instr:
            types[instr["dest"]] = instr["type"]
    return types


def func_to_ssa(func):
    blocks = block_map(form_blocks(func["instrs"]))
    add_entry(blocks)
    add_terminators(blocks)
    succ = {name: successors(block[-1]) for name, block in blocks.items()}
    dom = get_dom(succ, list(blocks.keys())[0])

    df = dom_fronts(dom, succ)
    defs = def_blocks(blocks)
    types = get_types(func)
    arg_names = {a["name"] for a in func["args"]} if "args" in func else set()

    gets = get_gets(blocks, df, defs)
    sets, get_dests, inits = ssa_rename(
        blocks, gets, succ, dom_tree(dom), arg_names
    )
    insert_sets_and_gets(blocks, sets, get_dests, types)
    insert_inits(next(iter(blocks.values())), inits, types)

    func["instrs"] = reassemble(blocks)


def to_ssa(bril):
    for func in bril["functions"]:
        func_to_ssa(func)
    return bril


if __name__ == "__main__":
    print(json.dumps(to_ssa(json.load(sys.stdin)), indent=2, sort_keys=True))
