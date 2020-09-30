import json
import sys
from collections import defaultdict

from cfg import block_map, successors, add_terminators
from form_blocks import form_blocks
from dom import get_dom, dom_fronts, dom_tree


def def_blocks(blocks):
    """Get a map from variable names to defining blocks.
    """
    out = defaultdict(set)
    for name, block in blocks.items():
        for instr in block:
            if 'dest' in instr:
                out[instr['dest']].add(name)
    return dict(out)


def get_phis(blocks, df, defs):
    """Find where to insert phi-nodes in the blocks.

    Produce a map from block names to variable names that need phi-nodes
    in those blocks. (We will need to generate names and actually insert
    instructions later.)
    """
    phis = {b: set() for b in blocks}
    for v, v_defs in defs.items():
        v_defs_list = list(v_defs)
        for d in v_defs_list:
            for block in df[d]:
                # Add a phi-node...
                if v not in phis[block]:
                    # ..unless we already did.
                    phis[block].add(v)
                    if block not in v_defs_list:
                        v_defs_list.append(block)
    return phis


def ssa_rename(blocks, phis, succ, domtree, vars):
    stack = {v: [] for v in vars}
    counters = {v: 0 for v in vars}
    phi_args = {b: {p: [] for p in phis[b]} for b in blocks}
    phi_dests = {b: {p: None for p in phis[b]} for b in blocks}

    def _push_fresh(var):
        fresh = '{}.{}'.format(var, counters[var])
        counters[var] += 1
        stack[var].insert(0, fresh)
        return fresh

    def _rename(block):
        # Save stacks.
        old_stack = {k: list(v) for k, v in stack.items()}

        # Rename phi-node destinations.
        for p in phis[block]:
            phi_dests[block][p] = _push_fresh(p)

        for instr in blocks[block]:
            # Rename arguments in normal instructions.
            if 'args' in instr:
                new_args = [stack[arg][0] for arg in instr['args']]
                instr['args'] = new_args

            # Rename destinations.
            if 'dest' in instr:
                instr['dest'] = _push_fresh(instr['dest'])

        # Rename phi-node arguments (in successors).
        for s in succ[block]:
            for p in phis[s]:
                if stack[p]:
                    phi_args[s][p].append((block, stack[p][0]))

        # Recursive calls.
        for b in domtree[block]:
            _rename(b)

        # Restore stacks.
        stack.update(old_stack)

    entry = list(blocks.keys())[0]
    _rename(entry)

    return phi_args, phi_dests


def insert_phis(blocks, phi_args, phi_dests, types):
    for block, instrs in blocks.items():
        for dest, pairs in phi_args[block].items():
            phi = {
                'op': 'phi',
                'dest': phi_dests[block][dest],
                'type': types[dest],
                'labels': [p[0] for p in pairs],
                'args': [p[1] for p in pairs],
            }
            instrs.insert(0, phi)


def reassemble(blocks):
    """Flatten a CFG into an instruction list."""
    # This could optimize slightly by opportunistically eliminating
    # `jmp .next` and `ret` terminators where it is allowed.
    instrs = []
    for name, block in blocks.items():
        instrs.append({'label': name})
        instrs += block
    return instrs


def get_types(blocks):
    # Silly way to get the type of variables. (According to the Bril
    # spec, well-formed programs must use only a single type for every
    # variable within a given function.)
    types = {}
    for block in blocks.values():
        for instr in block:
            if 'dest' in instr:
                types[instr['dest']] = instr['type']
    return types


def func_to_ssa(func):
    blocks = block_map(form_blocks(func['instrs']))
    add_terminators(blocks)
    succ = {name: successors(block[-1]) for name, block in blocks.items()}
    dom = get_dom(succ, list(blocks.keys())[0])

    df = dom_fronts(dom, succ)
    defs = def_blocks(blocks)
    types = get_types(blocks)

    phis = get_phis(blocks, df, defs)
    phi_args, phi_dests = ssa_rename(blocks, phis, succ,
                                     dom_tree(dom), set(defs.keys()))
    insert_phis(blocks, phi_args, phi_dests, types)

    func['instrs'] = reassemble(blocks)


def to_ssa(bril):
    for func in bril['functions']:
        func_to_ssa(func)
    return bril


if __name__ == '__main__':
    print(json.dumps(to_ssa(json.load(sys.stdin)), indent=2, sort_keys=True))
