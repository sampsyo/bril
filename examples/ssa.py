import sys
import json

from form_blocks import form_blocks, block_name
from mtm68_dom import find_doms, dom_frontier, doms_imm

def get_vars(blocks):
    """
    Returns a set of the variables used or defined in the given blocks.
    """
    varz = set()
    for block in blocks:
        for instr in block:
            if 'dest' in instr:
                varz.add(instr['dest'])
            # Make sure that we capture any undefined args
            if 'args' in instr:
                for arg in instr['args']:
                    varz.add(arg)
    return varz

def get_phi(block):
    pass

def get_defs(blocks, varz):
    """
    Returns a dictionary where keys are variables and values are the blocks
    where that variable is defined to.
    """
    defs = { v : set() for v in varz }
    for block in blocks:
        for instr in block:
            if 'dest' in instr:
                v = instr['dest']
                defs[v].add(block)
                continue # Don't bother searching rest of block
    return defs

def add_phi(block, var):
    block.insert(1, {
        'op' : 'phi'
        'dest' : dest,
        'type' : type,
        'labels' : labels,
        'args' : []
    })


def insert_phi(blocks, varz):
    defs = get_defs(blocks, varz)
    df = dom_frontier(func)
    for v in varz:
       for d in defs[v]:
           for block in df[d]:
               add_phi(block)
               add_def(defs, v)

def add_def(defs, v):
    pass

def fresh_dest():
    pass

def rename(block, stack, varz, cfg, doms_imm):
    names = []
    for instr in block:
        if 'args' in instr:
            pass
        if 'dest' in instr:
            dest = instr['dest']
            new_dest = fresh_dest()
            instr['dest'] = new_dest
            stack[dest].append(new_dest)
            names.append(new_dest)

    for succ in cfg.get_succ(block):
        for p in get_phi(succ):
            # TODO: stack[p]
            pass

    for b in doms_imm[block_name(block)]:
        rename(b stack, varz, cfg)

    for n in names:
        stack.pop(n)

def rename_blocks(blocks, varz):
    cfg = Cfg(blocks)
    stack = { v : [] for v in varz }
    entry = blocks[0]
    rename(entry, stack, varz, cfg )

def to_ssa(prog):
    for func in prog['functions']:
        blocks = list(form_blocks(func['instrs']))
        varz = get_vars(blocks)
        insert_phi(blocks, varz)
        rename_blocks(blocks, varz)

def from_ssa(prog):
    pass

def ssa(prog, arg):
    if arg == '-t':
        pass
    elif arg == '-f':
        pass
    else:
        print('Invalid argument: ' + arg)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: " + sys.argv[0] + " <-t | -f>")
    else:
        ssa(json.load(sys.stdin), sys.argv[1])
