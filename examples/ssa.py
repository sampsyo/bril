import json
import sys

from util import fresh
from collections import defaultdict
from cfg import block_map, block_successors, add_terminators
from dom import get_dom
from dom_frontier import get_frontiers
from form_blocks import form_blocks

def get_variable_definitions(blocks):
    """Given all blocks, return a map from variable name to the list of blocks 
    where that variable is defined
    """

    defns = defaultdict(set)
    for block_name, instrs in blocks.items():
        for instr in instrs:
            if 'dest' in instr:
                var = instr['dest']
                defns[var].add(block_name)
    return defns

def insert_phi_nodes(blocks, frontiers):
    var_defns = get_variable_definitions(blocks)

    queue = [(k, v) for k, v in var_defns.items()]

    while len(queue):
        var, defn_blocks = queue.pop(0)
        for defn_block in defn_blocks:
            for frontier_block in frontiers[defn_block]:
                instrs = blocks[frontier_block]

                # Check that we haven't already added this phi node
                if len(instrs) and instrs[0]['op'] == 'phi' \
                    and instrs[0]['dest'] == var:
                    instrs[0]['args'] += var
                    continue

                phi = {'dest' : var, 'op' : 'phi', 'args' : [var]}
                blocks[frontier_block] = [phi] + instrs

                if frontier_block not in var_defns[var]:
                    queue.append((var, [frontier_block]))

    return blocks

def rename(name, blocks, var_names, succ, dom_children, stacks):
    pushed = defaultdict(int)

    for instr in blocks[name]:
        # Replace arguments with new names
        if 'args' in instr and instr['op'] != 'phi':
            # print(instr['args'])
            instr['args'] = [stacks[v][-1] for v in instr['args'] if v in stacks]

        # Replace the destination with a fresh name
        if 'dest' in instr:
            old_name = instr['dest']
            fresh_name = fresh(old_name, var_names)
            var_names.append(fresh_name)
            instr['dest'] = fresh_name
            stacks[old_name].append(fresh_name)
            pushed[old_name] += 1 

    for succ_name in succ[name]:
        for instr in blocks[succ_name]:
            # Iterate only through phi blocks, which have been inserted at the
            # start of the block if they exist
            if instr['op'] != 'phi': break

            # Replace a single phi argument with the new stack value
            for i, arg in enumerate(instr['args']):
                if arg in stacks:
                    instr['args'][i] = stacks[arg][-1]
                    break

    for block in dom_children[name]:
        rename(block, blocks, var_names, succ, dom_children, stacks)
    
    for name, num_pushed in pushed.items():
        stacks[name] = stacks[name][:-num_pushed]

def print_blocks(blocks):
    for n, instrs in blocks.items():
        print(n)
        for i in instrs:
            print("\t", i)


def rename_all(blocks, succ, dom):
    var_names = list(get_variable_definitions(blocks).keys())
    stacks = {v : [v] for v in var_names}

    dom_children = defaultdict(set)
    for k, bls in dom.items():
        for b in bls:
            # We want direct children in the dominator tree only; so exclude the 
            # block itself and any non-direct successors
            if k != b:
                dom_children[b].add(k) 

    rename(next(iter(blocks)), blocks, var_names, succ, dom_children, stacks)
    print_blocks(blocks)

def to_ssa(bril):
    for func in bril['functions']:
        blocks = block_map(form_blocks(func['instrs']))
        add_terminators(blocks)
        succ = {name: block_successors(block) for name, block in blocks.items()}

        # dom maps each node to the nodes that dominate it
        dom = get_dom(succ, list(blocks.keys())[0])
        frontiers = get_frontiers(blocks, dom)
        blocks_with_phis = insert_phi_nodes(blocks, frontiers)

        # Remove self dominance
        for k, v in dom.items():
            v.remove(k)

        rename_all(blocks_with_phis, succ, dom)
        

if __name__ == '__main__':
    to_ssa(json.load(sys.stdin))