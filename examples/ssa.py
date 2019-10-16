import json
import sys

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

    to_add = defaultdict(set) #todo, check this
    
    for var, defn_blocks in var_defns.items():
        for defn_block in defn_blocks:
            for frontier_block in frontiers[defn_block]:
                instrs = blocks[frontier_block]

                # Check that we haven't already added this phi node
                if len(instrs) and instrs[0]['op'] == 'phi' \
                    and instrs[0]['dest'] == var:
                    continue

                phi = {'dest' : var, 'op' : 'phi'}
                blocks[frontier_block] = [phi] + instrs

                if frontier_block not in var_defns[var]:
                    to_add[var].add(frontier_block)





def to_ssa(bril):
    for func in bril['functions']:
        blocks = block_map(form_blocks(func['instrs']))
        add_terminators(blocks)
        succ = {name: block_successors(block) for name, block in blocks.items()}

        # dom maps each node to the nodes that dominate it
        dom = get_dom(succ, list(blocks.keys())[0])

        frontiers = get_frontiers(blocks, dom)
        insert_phi_nodes(blocks, frontiers)

        # frontiers = get_frontiers(blocks, dom)
        # print(frontiers)


if __name__ == '__main__':
    to_ssa(json.load(sys.stdin))