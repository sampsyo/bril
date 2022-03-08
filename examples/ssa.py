import sys
import json

from util import flatten
from form_blocks import form_blocks, block_name
from mtm68_dom import find_doms, dom_frontier, dom_imm
from mtm68_cfg import Cfg

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
    # Dest Instructions
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
    for instr in func['instrs']:
        if 'dest' in instr:
            typs[instr['dest']] = instr['type']
    return typs

def get_defs(blocks, varz):
    """
    Returns a dictionary where keys are variables and values are the
    block name corresponding where that variable is defined in.
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
    """
    Returns a dictionary where keys are blocks and values are
    variables corresponding to variables that need phi nodes
    in the given block.
    """
    defs = get_defs(blocks, varz)
    phis = { block_name(b) : set() for b in blocks }
    for v in varz:
        while defs[v]:
            d = defs[v].pop()
            for b in df[d]:
                phis[b].add(v)
                defs[v].add(b)
    return phis

def fresh_dest(v, fresh_counter):
    c = fresh_counter[v]
    fresh_counter[v] += 1
    return "{}.{}".format(v, c)

def rename(block, stack, varz, cfg, im_doms, fresh_counter,
           phis, phi_args, phi_dests):

    names = { v : 0 for v in varz }
    bname = block_name(block)

    # Rename phi node dests
    for p in phis[bname]:
        dest = fresh_dest(p, fresh_counter)
        phi_dests[bname][p] = dest
        stack[p].append(dest)
        names[p] += 1

    for instr in block:
        # Rename Args
        if 'args' in instr:
            instr['args'] = list(map(lambda a : stack[a][-1], instr['args']))
        # Rename Dests
        if 'dest' in instr:
            dest = instr['dest']
            new_dest = fresh_dest(dest, fresh_counter)
            instr['dest'] = new_dest
            stack[dest].append(new_dest)
            names[dest] += 1

        # Rename phi node args in succs
    for succ in cfg.get_succ(bname):
        sname = block_name(succ)
        for p in phis[sname]:
            phi_args[sname][p].append((stack[p][-1], sname))

    for b in im_doms[bname]:
        rename(cfg.get_block(b), stack, varz, cfg, im_doms, fresh_counter,
               phis, phi_args, phi_dests)

    for v, c in names.items():
        for i in range(c):
            stack[v].pop()

def rename_blocks(blocks, varz, im_doms, phis):
    """
    Modifies blocks so that they are using variables
    such that SSA is satisfied. Returns phi_args and phi_blocks
    which is a mapping from blocks to mapping of variables to args
    and dest respectivley (so that phi nodes can be injected into
    blocks).
    """
    cfg = Cfg(blocks)
    stack = { v : [v] for v in varz }
    entry = blocks[0]
    fresh_counter = { v : 0 for v in varz }

    phi_args = { block_name(b) : {p : []} for b in blocks
                                          for p in phis[block_name(b)]}

    phi_dests = { block_name(b) : {p : p} for b in blocks
                                          for p in phis[block_name(b)]}

    rename(entry, stack, varz, cfg, im_doms, fresh_counter,
           phis, phi_args, phi_dests)
    return phi_args, phi_dests


def insert_phis(phis, phi_args, phi_dests, types, blocks):
    for block in blocks:
        bname = block_name(block)
        for p in phis[bname]:
            phi = {
                'op'     : 'phi',
                'dest'   : phi_dests[bname][p],
                'type'   : types[p],
                'labels' : [a[1] for a in phi_args[bname][p]],
                'args'   : [a[0] for a in phi_args[bname][p]],
            }
            block.insert(1, phi)

def to_ssa(prog):
    for func in prog['functions']:
        blocks = list(form_blocks(func['instrs']))
        varz = get_vars(func)
        typs = get_typs(func) # Must be generated before renameing
        df = dom_frontier(func)

        phis = gen_phi(blocks, varz, df)

        im_doms = dom_imm(func)
        phi_args, phi_dests = rename_blocks(blocks, varz, im_doms, phis)

        insert_phis(phis, phi_args, phi_dests, typs, blocks)

        instrs = []
        for block in blocks:
            for instr in block:
                instrs.append(instr)
        func['instrs'] = instrs

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
