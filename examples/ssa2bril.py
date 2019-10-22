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


def replace_phis_with_copies(blocks):
    for name, instrs in blocks.items():
        num_phis = 0
        for instr in instrs:
            if instr['op'] != 'phi': break

            for arg, pred_name in zip(instr['args'], instr['sources']):
                pred_instrs = blocks[pred_name]

                copy_block = fresh('copy', blocks)
                # copy_instr = {
                #     'op' : 'id',
                #     'dest' : 
                # }

                terminator = pred_instrs[-1]['args']


                pred_instrs[-1]['args'] = [a if ]


            num_phis += 1
           

def from_ssa(bril): 
    for func in bril['functions']:
        blocks = block_map(form_blocks(func['instrs']))
        add_terminators(blocks)
        succ = {name: block_successors(block) for name, block in blocks.items()}




def ssa2bril():
    bril_without_phis = from_ssa(json.load(sys.stdin))
    print(bril_without_phis)

if __name__ == '__main__':
    ssa2bril()