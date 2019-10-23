import json
import sys

from collections import defaultdict

from cfg import block_map, block_successors, add_terminators
from dom import get_dom
from dom_frontier import get_frontiers
from form_blocks import form_blocks
from util import fresh, block_map_to_instrs


def replace_phis_with_copies(blocks):
    new_blocks = {}
    for current_block, instrs in blocks.items():
        num_phis = 0
        copy_block_names = {}
        for instr in instrs:
            if instr['op'] != 'phi': break
            num_phis += 1

            # Add a copy instruction per phi to the new block
            args = instr['args']
            numArgs = int(len(args)/2)
            for arg, pred_name in zip(args[:numArgs], args[numArgs:]):

                # Create a new block to hold the copies per source
                if pred_name not in copy_block_names:
                    names = list(blocks.keys()) + list(new_blocks.keys())
                    copy_block = fresh('copy', names)
                    copy_block_names[pred_name] = copy_block
                    new_blocks[copy_block] = []
                else:
                    copy_block = copy_block_names[pred_name]

                # Create a new copy instruction from this arg to the phi's dest
                copy_instr = {
                    'op' : 'id',
                    'dest' : instr['dest'],
                    'type' : instr['type'],
                    'args' : [arg],
                }

                new_blocks[copy_block].append(copy_instr)

        # Remove all the phi nodes from the current block
        del instrs[:num_phis]

        # Now, fix up the branches/jumps to link in the new blocks
        for source, copy_block in copy_block_names.items():
            # Now, fix up the branches/jumps to link in the new blocks
            # Jump from the copy to the current block.
            jmp_instr = {
                'op' : 'jmp',
                'args' : [current_block],
            }
            new_blocks[copy_block].append(jmp_instr)

            # Modify the source's terminator to go instead to the copy block
            terminator = blocks[source][-1]
            terminator['args'] = [copy_block if b == current_block else b for b in terminator['args']]
            
    blocks.update(new_blocks)

def from_ssa(bril): 
    for func in bril['functions']:
        blocks = block_map(form_blocks(func['instrs']))
        add_terminators(blocks)
        succ = {name: block_successors(block) for name, block in blocks.items()}

        replace_phis_with_copies(blocks)
        func['instrs'] = block_map_to_instrs(blocks)

    return json.dumps(bril, indent=4)

def ssa2bril():
    bril_without_phis = from_ssa(json.load(sys.stdin))
    print(bril_without_phis)

if __name__ == '__main__':
    ssa2bril()