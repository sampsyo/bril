import json
import sys

from collections import defaultdict

from cfg import block_map, block_successors, add_terminators
from dom import get_dom
from dom_frontier import get_dominator_tree
from form_blocks import form_blocks
from util import block_map_to_instrs

# from Briggs, Cooper, and Simpson, 1997.
def dominator_value_numbering(block_name, blocks, succ, dom_tree):
    # TODO intialize hash table scope

    for instr in blocks[block_name]:
        # Iterate only through phi nodes, which have been inserted at the
        # start of the block if they exist
        if instr['op'] != 'phi': break
        # TODO
        # meaningless?
        # redundant?
        pass

    # loop through all assignments 
    for instr in blocks[block_name]:
        # Skip phi nodes because we just processed them
        if 'dest' in instr and instr['op'] != 'phi': 
            # TODO
            pass

    # Iterate through all successor blocks' phi nodes
    for succ_name in succ[block_name]:
        for instr in blocks[succ_name]:
            if instr['op'] != 'phi': break
            # Replace phi arguments if they were calculated in this block
            for i, arg in enumerate(instr['args']):
                # TODO
                pass

    for child_name in dom_tree[block_name]:
        dominator_value_numbering(child_name, blocks, succ, dom_tree)

    # TODO clean up hash table for this scope



def global_value_numbering(bril): 
    for func in bril['functions']:
        blocks = block_map(form_blocks(func['instrs']))
        add_terminators(blocks)
        succ = {name: block_successors(block) for name, block in blocks.items()}

        dom_tree = get_dominator_tree(blocks, succ)

        func['instrs'] = block_map_to_instrs(blocks)

        dominator_value_numbering('next(iter(blocks))', blocks, succ, dom_tree)


    return json.dumps(bril, indent=4)

def gvn():
    bril = global_value_numbering(json.load(sys.stdin))
    #print(bril)

if __name__ == '__main__':
    gvn()