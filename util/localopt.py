"""Some examples demonstrating *local* optimization of Bril programs.
"""

import sys
import json
import itertools
from form_blocks import form_blocks, TERMINATORS


def flatten(ll):
    """Flatten an iterable of iterable to a single list.
    """
    return list(itertools.chain(*ll))


def var_args(instr):
    """Get a list of all the variables that are arguments to this
    instruction. If the instruction is "add x y", for example, return
    the list ["x", "y"]. Label arguments are not included.
    """
    if 'op' in instr:
        if instr['op'] == 'br':
            # Only the first argument to a branch is a variable.
            return [instr['args'][0]]
        elif instr['op'] in TERMINATORS:
            # Unconditional branch, for example, has no variable arguments.
            return []
        else:
            return instr.get('args', [])
    else:
        return []


def trivial_dce_pass(func):
    """Remove instructions from `func` that are never used as arguments
    to any other function. Return a bool indicating whether we deleted
    anything.
    """
    blocks = list(form_blocks(func['instrs']))

    # Find all the variables used as an argument to any instruction,
    # even once.
    used = set()
    for block in blocks:
        for instr in block:
            # Mark all the variable arguments as used.
            used.update(var_args(instr))

    # Delete the instructions that write to unused variables.
    changed = False
    for block in blocks:
        # Avoid deleting *effect instructions* that do not produce a
        # result. The `'dest' in i` predicate is true for all the *value
        # functions*, which are pure and can be eliminated if their
        # results are never used.
        new_block = [i for i in block
                     if 'dest' not in i or i['dest'] in used]

        # Record whether we deleted anything.
        changed |= len(new_block) != len(block)

        # Replace the block with the filtered one.
        block[:] = new_block

    # Reassemble the function.
    func['instrs'] = flatten(blocks)

    return changed


def trivial_dce(func):
    """Iteratively remove dead instructions, stopping when nothing
    remains to remove.
    """
    while trivial_dce_pass(func):
        pass


MODES = {
    'tdce': trivial_dce,
    'tdcep': trivial_dce_pass,
}


def localopt():
    if len(sys.argv) > 1:
        modify_func = MODES[sys.argv[1]]
    else:
        modify_func = trivial_dce

    # Apply the change to all the functions in the input program.
    bril = json.load(sys.stdin)
    for func in bril['functions']:
        modify_func(func)
    json.dump(bril, sys.stdout, indent=2, sort_keys=True)


if __name__ == '__main__':
    localopt()
