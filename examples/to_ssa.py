import json
import sys
from collections import defaultdict

from cfg import block_map, successors, add_terminators, add_entry, reassemble
from form_blocks import form_blocks
from dom import get_dom, dom_fronts, dom_tree, map_inv


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


def ssa_rename(blocks, phis, succ, domtree, args):
    stack = defaultdict(list, {v: [v] for v in args})
    phi_args = {b: {p: [] for p in phis[b]} for b in blocks}
    phi_dests = {b: {p: None for p in phis[b]} for b in blocks}
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
                if stack[p]:
                    phi_args[s][p].append((block, stack[p][0]))

        # Recursive calls.
        for b in sorted(domtree[block]):
            _rename(b)

        # Restore stacks.
        stack.update(old_stack)

    entry = list(blocks.keys())[0]
    _rename(entry)

    return phi_args, phi_dests


def prune_phis(pred, phi_args, phi_dests):
    """Prune possibly-undefined phi-nodes.

    Ordinary phi insertion will create phi-nodes that are "partially
    undefined" because they represent a convergence of paths where the
    variable is defined along some but not all paths. These phi-nodes
    are useless because it is illegal to read from the result. And they
    can confuse the out-of-SSA pass because it creates nonsensical
    copies. This algorithm iteratively eliminates such phi-nodes,
    propagating through to eliminate consumer phi-nodes until
    convergence.
    """
    # We build up a set of new names (phi destinations) to prune, and we
    # iterate until this set stops growing.
    old_prune_len = -1
    prune = set()
    while len(prune) != old_prune_len:
        old_prune_len = len(prune)

        # Look at each phi.
        for block, args in phi_args.items():
            dests = phi_dests[block]
            for v, v_args in args.items():
                # How many non-pruned arguments does this phi have?
                live_args = [a for _, a in v_args if a not in prune]
                if len(live_args) < len(pred[block]):
                    # Prune phis with insufficient arguments.
                    prune.add(dests[v])

    # Actually delete all phis with pruned destinations.
    for block, args in phi_args.items():
        dests = phi_dests[block]
        to_del = {v for v, d in dests.items() if d in prune}
        for v in to_del:
            del args[v]
            del dests[v]


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
    succ = {name: successors(block[-1]) for name, block in blocks.items()}
    pred = map_inv(succ)
    dom = get_dom(succ, list(blocks.keys())[0])

    df = dom_fronts(dom, succ)
    defs = def_blocks(blocks)
    types = get_types(func)
    arg_names = {a['name'] for a in func['args']} if 'args' in func else set()

    phis = get_phis(blocks, df, defs)
    phi_args, phi_dests = ssa_rename(blocks, phis, succ, dom_tree(dom),
                                     arg_names)
    prune_phis(pred, phi_args, phi_dests)
    insert_phis(blocks, phi_args, phi_dests, types)

    func['instrs'] = reassemble(blocks)


def to_ssa(bril):
    for func in bril['functions']:
        func_to_ssa(func)
    return bril


if __name__ == '__main__':
    print(json.dumps(to_ssa(json.load(sys.stdin)), indent=2, sort_keys=True))
