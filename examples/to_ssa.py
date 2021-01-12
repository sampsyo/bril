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


def type_is_ptr(var_type):
    return type(var_type) == dict


def type_str(var_type):
    if type_is_ptr(var_type):
        ptr_type = var_type["ptr"]
        return "ptr." + type_str(ptr_type)
    assert type(var_type) == str
    return var_type


def undefined_var(var_type):
    return "__undefined." + type_str(var_type)


def ssa_rename(blocks, phis, succ, domtree, args, types):
    stack = {v: [v] for v in args}
    phi_args = {b: {p: [] for p in phis[b]} for b in blocks}
    phi_dests = {b: {p: None for p in phis[b]} for b in blocks}
    counters = defaultdict(int)

    def _push_fresh(stack, var):
        fresh = '{}.{}'.format(var, counters[var])
        counters[var] += 1
        stack[var].insert(0, fresh)
        return fresh

    def _rename(block, stack):
        # Copy the stack so the callers stack isn't mutated.
        stack = defaultdict(list, {v: list(s) for v, s in stack.items()})

        # Rename phi-node destinations.
        for p in phis[block]:
            phi_dests[block][p] = _push_fresh(stack, p)

        for instr in blocks[block]:
            # Rename arguments in normal instructions.
            if 'args' in instr:
                new_args = [stack[arg][0] for arg in instr['args']]
                instr['args'] = new_args

            # Rename destinations.
            if 'dest' in instr:
                instr['dest'] = _push_fresh(stack, instr['dest'])

        # Rename phi-node arguments (in successors).
        for s in succ[block]:
            for p in phis[s]:
                if stack[p]:
                    phi_args[s][p].append((block, stack[p][0]))
                else:
                    # The variable is not defined on this path
                    phi_args[s][p].append((block, undefined_var(types[p])))

        # Recursive calls.
        for b in sorted(domtree[block]):
            _rename(b, stack)

    entry = list(blocks.keys())[0]
    _rename(entry, stack)

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


def undefined_value_instr(var_type):
    if type_is_ptr(var_type):
        # The memory bril extension doesn't have a null
        # pointer value, so allocate a 0 sized region,
        # which we will never free.
        ptr_type = var_type["ptr"]
        return {
            'op': 'alloc',
            'dest': undefined_var(var_type),
            'type': var_type,
            'args': ['__undefined.zero'],
        }
    else:
        values = {
            "int": 0,
            "bool": False,
            "float": 0.0,
        }
        return {
            'op': 'const',
            'dest': undefined_var(var_type),
            'type': var_type,
            'value': values[var_type],
        }


def get_unique_types(types):
    unique_types = {type_str(t): t for t in types.values()}
    unique_types = sorted(unique_types.items(), key=lambda t: t[0])
    return [t for _, t in unique_types]


def insert_undefined_vars(blocks, types):
    entry = list(blocks.keys())[0]
    unique_types = get_unique_types(types)

    for var_type in unique_types:
        instr = undefined_value_instr(var_type)
        blocks[entry].insert(0, instr)

    need_zero = any({type_is_ptr(t) for t in unique_types})
    if need_zero:
        zero = {
            'op': 'const',
            'dest': '__undefined.zero',
            'type': 'int',
            'value': 0,
        }
        blocks[entry].insert(0, zero)


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
                                     arg_names, types)
    insert_phis(blocks, phi_args, phi_dests, types)
    insert_undefined_vars(blocks, types)

    func['instrs'] = reassemble(blocks)


def to_ssa(bril):
    for func in bril['functions']:
        func_to_ssa(func)
    return bril


if __name__ == '__main__':
    print(json.dumps(to_ssa(json.load(sys.stdin)), indent=2, sort_keys=True))
