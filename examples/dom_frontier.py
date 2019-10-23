import json
import sys

from collections import defaultdict

from cfg import block_map, block_successors, add_terminators
from dom import get_dom
from form_blocks import form_blocks

# Takes in a block and its dominated blocks, returns blocks on the frontier (not
# dominated, but almost!)
def get_frontiers(blocks, dom):

    frontiers = {v: set() for v in blocks}

    for name, instrs in blocks.items():
        # Blocks that dominate us
        dominators = dom[name]

        # Check if successors are not dominatd by the same blocks
        for succ in block_successors(instrs):
            succ_dominators = dom[succ]

            for d in dominators:
                if d not in succ_dominators:
                    frontiers[d].add(succ)

    return frontiers

def get_doms(bril):
    for func in bril['functions']:
        blocks = block_map(form_blocks(func['instrs']))
        add_terminators(blocks)
        succ = {name: block_successors(block) for name, block in blocks.items()}

        # dom maps each node to the nodes that dominate it
        dom = get_dom(succ, list(blocks.keys())[0])

        frontiers = get_frontiers(blocks, dom)
        print(frontiers)


def get_dominator_tree(blocks, succ):
    dom = get_dom(succ, list(blocks.keys())[0])

    tree = defaultdict(set)
    for k, bls in dom.items():
        for b in bls:
            # Exclude the block itself 
            if k != b:
                tree[b].add(k) 

    block_names = list(blocks.keys())

    # Remove redudant edges to get the dominator tree
    for i in block_names:
        for j in block_names:
            if j in tree[i]:
                for k in block_names:
                    if k in tree[j]:
                        tree[i].remove(k)
    return tree

if __name__ == '__main__':
    get_doms(json.load(sys.stdin))