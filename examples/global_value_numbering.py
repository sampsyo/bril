import json
import sys

from collections import defaultdict, namedtuple
from functools import reduce, partial

from cfg import block_map, block_successors, add_terminators
from dom import get_dom
from dom_frontier import get_dominator_tree
from form_blocks import form_blocks
from util import block_map_to_instrs

# From Adrian's implementation of local value numbering
Value = namedtuple('Value', ['op', 'args'])
def canonicalize(instr):
    value = Value(instr['op'], tuple(instr['args']))
    if value.op in ('add', 'mul'):
        return Value(value.op, tuple(sorted(value.args)))
    else:
        return value

def reverse_post_order(start_node, succ):
  def explore(path, visited, node):
    if (node in visited) or (node in path):
      return visited
    else:
      new_path = [node] + path
      edges = succ[node]

      visited = reduce(partial(explore, new_path), edges, visited)
      return [node] + visited
  return explore([], [], start_node)

def check_removeable_phi(instr, vars_to_value_nums, exprs_to_value_nums):
    # Meaningless phi nodes are those whose arguments all already have
    # the same value number
    canonicalized = canonicalize(instr)
    meaningless = True
    arg_value_number = None

    args = instr['args']
    numArgs = int(len(args)/2)
    for i in range(numArgs):
        arg = args[i]
        if arg not in vars_to_value_nums:
            meaningless = False
            break
        if not arg_value_number:
            arg_value_number = vars_to_value_nums[arg]
            continue
        if arg_value_number != vars_to_value_nums[arg]:
            meaningless = False
            break

    if meaningless:
        # Map the destination to just the existing arg value number
        vars_to_value_nums[instr['dest']] = arg_value_number
        return True

    # Redundant phi nodes are those that are identical to ones we 
    # already computed: those in the hash table
    redundant = canonicalized in exprs_to_value_nums and len(exprs_to_value_nums[canonicalized])
    if redundant:
        # Use the previous value number we already calculated
        value_number = exprs_to_value_nums[canonicalized][-1]
        vars_to_value_nums[instr['dest']] = value_number
        return True
    
    return False

FOLDABLE_OPS = {
    'add': lambda a, b: a + b,
    'mul': lambda a, b: a * b,
    'sub': lambda a, b: a - b,
    'div': lambda a, b: a // b,
    'gt': lambda a, b: a > b,
    'lt': lambda a, b: a < b,
    'ge': lambda a, b: a >= b,
    'le': lambda a, b: a <= b,
}

# From Briggs, Cooper, and Simpson, 1997.
def dominator_value_numbering(func, blocks, succ, dom_tree, reverse_post_order):

    # Maps variable names to value numbers (in this case, also names)
    vars_to_value_nums = {arg['name'] : arg['name'] for arg in func['args']}

    # The core "hashtable" for this algorithm. Maps expressions to value 
    # numbers.
    exprs_to_value_nums = defaultdict(list)

    # Map from value numbers to constants for constant propagation/folding
    value_nums_to_consts = {}

    def dvn(block_name):
        # Intialize hashtable scope (track changes to be cleaned after this
        # recursive call)
        pushed = defaultdict(int)
        instrs_to_remove = []

        for instr in blocks[block_name]:
            # Iterate only through phi nodes, which have been inserted at the
            # start of the block if they exist
            if instr['op'] != 'phi': break
            
            canonicalized = canonicalize(instr)

            if check_removeable_phi(instr, vars_to_value_nums, exprs_to_value_nums):
                instrs_to_remove.append(instr)
            else:
                vars_to_value_nums[instr['dest']] = instr['dest']

                exprs_to_value_nums[canonicalized].append(instr['dest']) 
                pushed[canonicalized] += 1 

        # Loop through all assignments 
        for instr in blocks[block_name]:

            # Add constants to our mapping
            if instr['op'] == 'const':
                value_nums_to_consts[instr['dest']] = instr['value']

            # Skip phi nodes because we just processed them
            if 'dest' in instr and 'args' in instr and instr['op'] != 'phi':

                instr['args'] = [vars_to_value_nums[a] if a in vars_to_value_nums else a for a in instr['args']]
                
                canonicalized = canonicalize(instr)
                value_number = None

                # Check if we already have a value number for this expression
                if canonicalized in exprs_to_value_nums and len(exprs_to_value_nums[canonicalized]):
                    value_number = exprs_to_value_nums[canonicalized][-1]
                    instrs_to_remove.append(instr)
                # Basic copy propagation by looking up the first argument
                elif instr['op'] == 'id' and instr['args'][0] in vars_to_value_nums:
                    value_number = vars_to_value_nums[instr['args'][0]]
                    instrs_to_remove.append(instr)
                else:
                    # Add a "new" value number to our data structures
                    value_number = instr['dest']
                    vars_to_value_nums[instr['dest']] = instr['dest']
                    exprs_to_value_nums[canonicalized].append(instr['dest'])
                    pushed[canonicalized] += 1

                    # Constant folding if possible
                    if instr['op'] in FOLDABLE_OPS:
                        const_args = [value_nums_to_consts[n] for n in instr['args'] if n in value_nums_to_consts]
                        if len(const_args) == len(instr['args']):
                            value = FOLDABLE_OPS[instr['op']](*const_args)
                            instr.update({
                                'op': 'const',
                                'value': value,
                            })
                            del instr['args']
                            value_nums_to_consts[value_number] = value

                vars_to_value_nums[instr['dest']] = value_number

            # Replace arguments to non-phis as needed
            elif 'args' in instr and instr['op'] != 'phi':
                for i, arg in enumerate(instr['args']):
                    if arg in vars_to_value_nums:
                        instr['args'][i] = vars_to_value_nums[arg]

        # Iterate through all successor blocks' phi nodes
        for succ_name in succ[block_name]:
            for instr in blocks[succ_name]:
                if instr['op'] != 'phi': break
                # Replace phi arguments if they were calculated in this block
                args = instr['args']
                numArgs = int(len(args)/2)
                for i, (arg, pred_name) in enumerate(zip(args[:numArgs], args[numArgs:])):
                    if pred_name != block_name or arg not in vars_to_value_nums:
                        continue

                    instr['args'][i] = vars_to_value_nums[arg]


        for instr in instrs_to_remove:
            blocks[block_name].remove(instr)

        ordered_children = [b for b in reverse_post_order if b in dom_tree[block_name]]
        for child_name in ordered_children:
            dvn(child_name)

        # Clean up hashtable for this scope
        for expr, num_pushed in pushed.items():
            exprs_to_value_nums[expr] = exprs_to_value_nums[expr][:-num_pushed]

    # Call recursive helper on entry block
    dvn(next(iter(blocks)))

def global_value_numbering(bril): 
    for func in bril['functions']:
        blocks = block_map(form_blocks(func['instrs']))
        add_terminators(blocks)
        succ = {name: block_successors(block) for name, block in blocks.items()}

        dom_tree = get_dominator_tree(blocks, succ)
        order = reverse_post_order(next(iter(blocks)), succ)

        dominator_value_numbering(func, blocks, succ, dom_tree, order)
        func['instrs'] = block_map_to_instrs(blocks)

    return json.dumps(bril, indent=4)

def gvn():
    bril = global_value_numbering(json.load(sys.stdin))
    print(bril)

if __name__ == '__main__':
    gvn()