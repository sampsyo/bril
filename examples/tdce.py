"""Trivial dead code elimination for Bril programs---a demonstration of
local optimization.
"""

import sys
import json
from form_blocks import form_blocks
from util import flatten, var_args


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
    # An exercise for the reader: prove that this loop terminates.
    while trivial_dce_pass(func):
        pass


def drop_killed_local(block):
    """Delete instructions in a single block whose result is unused
    before the next assignment. Return a bool indicating whether
    anything changed.
    """
    # A map from variable names to the last place they were assigned
    # since the last use. These are candidates for deletion---if a
    # variable is assigned while in this map, we'll delete what the maps
    # point to.
    last_def = {}

    # Find the indices of droppable instructions.
    to_drop = set()
    for i, instr in enumerate(block):
        # Check for uses. Anything we use is no longer a candidate for
        # deletion.
        for var in var_args(instr):
            if var in last_def:
                del last_def[var]

        # Check for definitions. This *has* to happen after the use
        # check, so we don't count "a = a + 1" as killing a before using
        # it.
        if 'dest' in instr:
            dest = instr['dest']
            if dest in last_def:
                # Another definition since the most recent use. Drop the
                # last definition.
                to_drop.add(last_def[dest])
            last_def[dest] = i

    # Remove the instructions marked for deletion.
    new_block = [instr for i, instr in enumerate(block)
                 if i not in to_drop]
    changed = len(new_block) != len(block)
    block[:] = new_block
    return changed


def drop_killed_pass(func):
    """Drop killed functions from *all* blocks. Return a bool indicating
    whether anything changed.
    """
    blocks = list(form_blocks(func['instrs']))
    changed = False
    for block in blocks:
        changed |= drop_killed_local(block)
    func['instrs'] = flatten(blocks)
    return changed


def trivial_dce_plus(func):
    """Like `trivial_dce`, but also deletes locally killed instructions.
    """
    while trivial_dce_pass(func) or drop_killed_pass(func):
        pass


MODES = {
    'tdce': trivial_dce,
    'tdcep': trivial_dce_pass,
    'dkp': drop_killed_pass,
    'tdce+': trivial_dce_plus,
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
