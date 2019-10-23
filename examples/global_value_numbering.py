import json
import sys

from collections import defaultdict, namedtuple

from cfg import block_map, block_successors, add_terminators
from dom import get_dom
from dom_frontier import get_dominator_tree
from form_blocks import form_blocks
from util import block_map_to_instrs

# from Adrian's implementation of local value numbering
Value = namedtuple('Value', ['op', 'args'])
def canonicalize(instr):
    value = Value(instr['op'], tuple(instr['args']))
    if value.op in ('add', 'mul'):
        return Value(value.op, tuple(sorted(value.args)))
    else:
        return value


# From Briggs, Cooper, and Simpson, 1997.
def dominator_value_numbering(block_name, blocks, succ, dom_tree):

    # Maps variable names to value numbers (in this case, also names)
    vars_to_value_nums = {}

    # The core "hashtable" for this algorithm. Maps expressions to value 
    # numbers.
    exprs_to_value_nums = defaultdict(list)

    def dvn(block_name):

        # Intialize hashtable scope (track changes to be cleaned after this
        # recursive call)
        pushed = defaultdict(int)
        instrs_to_remove = []

        for instr in blocks[block_name]:
            # Iterate only through phi nodes, which have been inserted at the
            # start of the block if they exist
            if instr['op'] != 'phi': break
            
            # TODO: verify that it's okay to discard the destination when
            # canonicalizing phi nodes
            canonicalized = canonicalize(instr)

            # Meaningless phi nodes are those whose arguments all already have
            # the same value number
            meaningless = True
            arg_value_number = None
            for arg in instr['args']:
                if arg not in vars_to_value_nums:
                    meaningless = False
                    break
                if not arg_value_number:
                    arg_value_number = vars_to_value_nums[arg]
                    continue
                if arg_value_number != vars_to_value_nums[arg]:
                    meaningless = False
                    break

            # Redundant phi nodes are those that are identical to ones we 
            # already computed: those in the hash table
            redundant = canonicalized in exprs_to_value_nums

            if meaningless:
                # Map the destination to just the existing arg value number
                vars_to_value_nums[instr['dest']] = arg_value_number

                instrs_to_remove.append(instr)
            elif redundant:
                # Use the previous value number we already calculated
                value_number = exprs_to_value_nums[canonicalized][-1]
                vars_to_value_nums[instr['dest']] = value_number

                instrs_to_remove.append(instr)
            else:
                vars_to_value_nums[instr['dest']] = instr['dest']

                exprs_to_value_nums[canonicalized].append(instr['dest']) 
                pushed[canonicalized] += 1 

        # Loop through all assignments 
        for instr in blocks[block_name]:
            # Skip phi nodes because we just processed them
            if 'dest' in instr and 'args' in instr and instr['op'] != 'phi': 

                instr['args'] = [vars_to_value_nums[a] if a in vars_to_value_nums else a for a in instr['args']]
                
                canonicalized = canonicalize(instr)

                # TODO: maybe "simplify" expr

                # Check if we already have a vlaue number for this expression
                if canonicalized in exprs_to_value_nums:
                    value_number = exprs_to_value_nums[canonicalized][-1]
                    vars_to_value_nums[instr['dest']] = value_number
                    instrs_to_remove.append(instr)
                else:
                    vars_to_value_nums[instr['dest']] = instr['dest']
                    exprs_to_value_nums[canonicalized].append(instr['dest'])
                    pushed[canonicalized] += 1

        # Iterate through all successor blocks' phi nodes
        for succ_name in succ[block_name]:
            for instr in blocks[succ_name]:
                if instr['op'] != 'phi': break
                # Replace phi arguments if they were calculated in this block
                for i, arg in enumerate(instr['args']):
                    # TODO
                    pass

        for child_name in dom_tree[block_name]:
            dvn(child_name)

        # Clean up hashtable for this scope
        for expr, num_pushed in pushed.items():
            exprs_to_value_nums[expr] = exprs_to_value_nums[expr][:-num_pushed]

    # Call recursive helper
    dvn(block_name)

def global_value_numbering(bril): 
    for func in bril['functions']:
        blocks = block_map(form_blocks(func['instrs']))
        add_terminators(blocks)
        succ = {name: block_successors(block) for name, block in blocks.items()}

        dom_tree = get_dominator_tree(blocks, succ)

        func['instrs'] = block_map_to_instrs(blocks)

        dominator_value_numbering(next(iter(blocks)), blocks, succ, dom_tree)


    return json.dumps(bril, indent=4)

def gvn():
    bril = global_value_numbering(json.load(sys.stdin))
    #print(bril)

if __name__ == '__main__':
    gvn()