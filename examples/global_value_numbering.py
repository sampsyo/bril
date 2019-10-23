import json
import sys

from collections import defaultdict

from cfg import block_map, block_successors, add_terminators
from dom import get_dom
from dom_frontier import get_dominator_tree
from form_blocks import form_blocks
from util import block_map_to_instrs

def global_value_numbering(bril): 
    for func in bril['functions']:
        blocks = block_map(form_blocks(func['instrs']))
        add_terminators(blocks)
        succ = {name: block_successors(block) for name, block in blocks.items()}

        dom_tree = get_dominator_tree(blocks, succ)

        func['instrs'] = block_map_to_instrs(blocks)

    return json.dumps(bril, indent=4)

def gvn():
    bril = global_value_numbering(json.load(sys.stdin))
    print(bril)

if __name__ == '__main__':
    gvn()