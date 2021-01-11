import json
import sys
from collections import defaultdict

from cfg import block_map, successors, add_terminators, add_entry, reassemble, edges
from form_blocks import form_blocks
from dom import get_dom, dom_fronts, dom_tree, map_inv


def reaching_definitions(init, blocks, pred, succ):
    """
    Reaching definitions analysis of `blocks`. Returns a
    mapping for each the `in` and `out` of a block, in the
    following manner:

    `block` -> {v1, v2, ...}
    where `v1, v2, ...` are reached in `block`.
    """

    def merge(_initial, _pred, _out):
        return set().union(_initial, *[_out[name] for name in _pred])

    def transfer(_in, _block):
        return _in.symmetric_difference(definitions[_block])

    # Initialize.
    block_names = list(blocks.keys())
    in_ = {name: {} for name in block_names}
    out_ = {name: {} for name in block_names}

    entry = block_names[0]
    for a in init:
        in_[entry][a] = {entry}

    # Mapping from block name to a set of definitions within the block.
    definitions = {n: set(i['dest'] for i in blocks[n] if 'dest' in i)
                   for n in block_names}

    worklist = [name for name in block_names]
    while worklist:
        b = worklist.pop()
        in_[b] = merge(in_[b], pred[b], out_)
        copy = out_[b].copy()
        out_[b] = transfer(in_[b], b)
        if out_[b] == copy:
            continue
        worklist.extend(succ[b])
    return in_, out_, definitions


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


def ssa_rename(blocks, phis, pred, succ, domtree, args):
    stack = defaultdict(list, {v: [v] for v in args})
    phi_args = {b: {p: [] for p in phis[b]} for b in blocks}
    phi_dests = {b: {p: None for p in phis[b]} for b in blocks}
    in_, out_, definitions = reaching_definitions(args, blocks, pred, succ)
    counters = defaultdict(int)

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
                # We only want to add a variable `v` to phi if
                # (1) `v` has already been defined along the predecessors path to `s`,
                # (2) `v` is defined in the current block, or
                # (3) `v` is defined in the incoming definitions of the block.
                is_defined = any(p in out_[b] for b in pred[block]) \
                             or p in definitions[block] \
                             or p in in_[block]

                if is_defined and stack[p]:
                    phi_args[s][p].append((block, stack[p][0]))
                else:
                    # The variable is not defined on this path
                    phi_args[s][p].append((block, "__undefined"))

        # Recursive calls.
        for b in sorted(domtree[block]):
            _rename(b)

        # Restore stacks.
        stack.update(old_stack)

    entry = list(blocks.keys())[0]
    _rename(entry)

    return phi_args, phi_dests


def insert_phis(blocks, phi_args, phi_dests, types):
    for block, instrs in blocks.items():
        for dest, pairs in sorted(phi_args[block].items()):
            phi = {
                'op': 'phi',
                'dest': phi_dests[block][dest],
                'type': types[dest],
                'labels': [p[0] for p in pairs],
                'args': [p[1] for p in pairs],
            }
            instrs.insert(0, phi)


def get_types(func):
    # Silly way to get the type of variables. (According to the Bril
    # spec, well-formed programs must use only a single type for every
    # variable within a given function.)
    types = {arg['name']: arg['type'] for arg in func.get('args', [])}
    for instr in func['instrs']:
        if 'dest' in instr:
            types[instr['dest']] = instr['type']
    return types


def func_to_ssa(func):
    blocks = block_map(form_blocks(func['instrs']))
    add_entry(blocks)
    add_terminators(blocks)
    pred, succ = edges(blocks)
    dom = get_dom(succ, list(blocks.keys())[0])

    df = dom_fronts(dom, succ)
    defs = def_blocks(blocks)
    types = get_types(func)
    arg_names = {a['name'] for a in func['args']} if 'args' in func else set()

    phis = get_phis(blocks, df, defs)
    phi_args, phi_dests = ssa_rename(blocks, phis, pred, succ,
                                     dom_tree(dom), arg_names)
    insert_phis(blocks, phi_args, phi_dests, types)

    func['instrs'] = reassemble(blocks)


def to_ssa(bril):
    for func in bril['functions']:
        func_to_ssa(func)
    return bril


if __name__ == '__main__':
    print(json.dumps(to_ssa(json.load(sys.stdin)), indent=2, sort_keys=True))
