import sys
import json

from form_blocks import form_blocks, block_name
from mtm68_dom import find_doms, dom_frontier, doms_imm

def get_vars(func):
    """
    Returns a set of the variables that are arguments to the function or
    defined in the given blocks.
    """
    varz = set()
    # Function Arguments
    if 'args' in func:
        for arg in func['args']:
            varz.add(arg['name'])
    # Dest Arguments
    for instr in func['instrs']:
        if 'dest' in instr:
            varz.add(instr['dest'])
    return varz

def get_typs(func):
    """
    Returns a dict of the variables to types in the given blocks
    """
    typs = {}
    # Function Arguments
    if 'args' in func:
        for arg in func['args']:
            typs[arg['name']] = arg['type']
    # Dest Instructions
    for block in blocks:
        for instr in block:
            if 'dest' in instr:
                typs[instr['dest']] = instr['type']
    return typs

def get_defs(blocks, varz):
    """
    Returns a dictionary where keys are variables and values are the
    block names corresponding where that variable is defined in.
    """
    defs = { v : set() for v in varz }
    for block in blocks:
        for instr in block:
            if 'dest' in instr:
                v = instr['dest']
                defs[v].add(block_name(block))
                continue # Don't bother searching rest of block
    return defs

def gen_phi(blocks, varz, df):
    defs = get_defs(blocks, varz)
    phis = { b : set() for b in blocks }
    for v in varz:
       for d in defs[v]:
           for block in df[d]:
               phis[block].add(v)
               defs[v].add(block)
    return phis

def fresh_dest(v, fresh_counter):
    fresh_counter[v] += 1
    c = fresh_counter[v]
    return "{}.{}".format(v, c)

def rename(block, stack, varz, cfg, im_doms, fresh_counter,
           phis, phi_args, phi_dests):
    names = {}
    for instr in block:
        # Rename Args
        if 'args' in instr:
            instr['args'] = map(lambda a : stack[a][-1], instr['args'])
        # Rename Dests
        if 'dest' in instr:
            dest = instr['dest']
            new_dest = fresh_dest(dest, fresh_counter)
            instr['dest'] = new_dest
            stack[dest].append(new_dest)
            names[dest].append(new_dest)

    # Rename phi node dests
    for p in phis[block]:
        phi_dests[block][p] = fresh_dest(p, fresh_counter)

    # Rename phi node args in succs
    for succ in cfg.get_succ(block):
        for p in phis[succ]:
            phi_args[s][p].append((stack[p][-1]), block_name(succ))

    for b in im_doms[block_name(block)]:
        rename(b, stack, varz, cfg, fresh_counter,
               phis, phi_args, phi_dests)

    for v, ns in names.items():
        for n in ns:
            stack[v].pop(n)

def rename_blocks(blocks, varz, im_doms, phis):
    cfg = Cfg(blocks)
    stack = { v : [v] for v in varz }
    entry = blocks[0]
    fresh_counter = { v : 0 for v in varz }
    phi_args = { b: {p : []} for b in blocks for p in phis[b]}
    phi_dests = { b : {p : p} for b in blocks for p in phis[b]}
    rename(entry, stack, varz, cfg, im_doms, fresh_counter,
           phis, phi_args, phi_dests)
    return phi_args, phi_dests


def insert_phis(phis, phi_args, phi_dests, types, blocks):
    for block in blocks:
        for p in phis[block]:
            phi = {
                'op'     : 'phi',
                'dest'   : phi_dests[block][p],
                'type'   : types[phi_dests[block][p]],
                'labels' : [a[1] for a in phi_args[block][p]],
                'args'   : [a[0] for a in phi_args[block][p]],
            }
            block['instrs'].insert(0, phi)

def to_ssa(prog):
    for func in prog['functions']:
        blocks = list(form_blocks(func['instrs']))
        varz = get_vars(func)
        df = dom_frontier(func)
        phis = gen_phi(blocks, varz, df)
        im_doms = doms_imm(func)
        phi_args, phi_dests = rename_blocks(blocks, varz, im_doms, phis)
        typs = get_typs(func)
        insert_phi(phis, phi_args, phi_dests, typs, blocks)
        func['instrs'] = flatten(blocks)

def from_ssa(prog):
    pass

def ssa(prog, arg):
    if arg == '-t':
        to_ssa(prog)
        json.dump(prog, sys.stdout, indent=2, sort_keys=True)
    elif arg == '-f':
        pass
    else:
        print('Invalid argument: ' + arg)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: " + sys.argv[0] + " <-t | -f>")
    else:
        ssa(json.load(sys.stdin), sys.argv[1])
