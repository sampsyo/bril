import json
import operator
import sys

from collections import defaultdict
from functools import reduce

from cfg import block_map, block_successors, add_terminators
from dom import get_dom
from dom_frontier import get_frontiers
from form_blocks import form_blocks
from util import fresh

def get_variable_definitions(blocks):
    """Given all blocks, return a map from variable name to a set of
    the blocks where that variable is defined and a map from variable name to
    its type
    """

    defns = defaultdict(set)
    types = {}
    for block_name, instrs in blocks.items():
        for instr in instrs:
            if 'dest' in instr:
                var = instr['dest']
                defns[var].add((block_name))
                types[var] = instr['type']
    return defns, types

def insert_phi_nodes(blocks, frontiers, preds):
    var_defns, types = get_variable_definitions(blocks)

    queue = [(k, v) for k, v in var_defns.items()]

    while len(queue):
        var, defn_blocks = queue.pop(0)
        for defn_block in defn_blocks:
            for frontier_block in frontiers[defn_block]:
                instrs = blocks[frontier_block]

                # Check that we haven't already added this phi node
                if len(instrs) and instrs[0]['op'] == 'phi' \
                    and instrs[0]['dest'] == var:
                    continue

                # Number of args to phi is the number of predecessors of this 
                # block
                num_preds = len(preds[frontier_block])
                phi = {
                  'dest' : var,
                  'type': types[var],
                  'op' : 'phi',
                  'args' : [var] * num_preds,
                  'sources': [""] * num_preds,
                }

                blocks[frontier_block] = [phi] + instrs

                if frontier_block not in var_defns[var]:
                    queue.append((var, [frontier_block]))

    return blocks

def rename(name, blocks, var_names, succ, dom_children, stacks):
    pushed = defaultdict(int)

    for instr in blocks[name]:
        # Replace arguments with new names
        if 'args' in instr:
            instr['args'] = [stacks[v][-1] if v in stacks else v for v in instr['args']]

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
                    instr['sources'][i] = name
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
    var_defns, _ = get_variable_definitions(blocks)
    var_names = list(var_defns.keys())
    stacks = {v : [v] for v in var_names}

    dom_children = defaultdict(set)
    for k, bls in dom.items():
        for b in bls:
            # We want direct children in the dominator tree only; so exclude the 
            # block itself 
            if k != b:
                dom_children[b].add(k) 

    block_names = list(blocks.keys())

    # Remove redudant edges to get the dominator tree
    for i in block_names:
        for j in block_names:
            if j in dom_children[i]:
                for k in block_names:
                    if k in dom_children[j]:
                        dom_children[i].remove(k)

    print(dom_children)
    rename(next(iter(blocks)), blocks, var_names, succ, dom_children, stacks)

def to_ssa(bril):
    for func in bril['functions']:
        blocks = block_map(form_blocks(func['instrs']))
        add_terminators(blocks)
        succ = {name: block_successors(block) for name, block in blocks.items()}

        # dom maps each node to the nodes that dominate it
        dom = get_dom(succ, list(blocks.keys())[0])
        frontiers = get_frontiers(blocks, dom)

        preds = defaultdict(set)
        for k, values in succ.items():
            for v in values:
                preds[v].add(k)

        blocks_with_phis = insert_phi_nodes(blocks, frontiers, preds)

        # Remove self dominance
        for k, v in dom.items():
            v.remove(k)

        rename_all(blocks_with_phis, succ, dom)
        func['instrs'] = reduce(operator.iconcat, [[{'label' : k}] + v for k, v in blocks_with_phis.items()], [])

    return json.dumps(bril, indent=4)

def bril2ssa():
    ssa = to_ssa(json.load(sys.stdin))
    print(ssa)

if __name__ == '__main__':
    bril2ssa()