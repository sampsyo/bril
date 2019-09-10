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
    blocks = list(form_blocks(func['instrs']))

    # Find all the variables used as an argument to any instruction,
    # even once.
    used = set()
    for block in blocks:
        for instr in block:
            # Mark all the variable arguments as used.
            used.update(var_args(instr))

    # Delete the instructions that write to unused variables.
    for block in blocks:
        # Avoid deleting *effect instructions* that do not produce a
        # result. The `'dest' in i` predicate is true for all the *value
        # functions*, which are pure and can be eliminated if their
        # results are never used.
        block[:] = [i for i in block
                    if 'dest' not in i or i['dest'] in used]

    # Reassemble the function.
    func['instrs'] = flatten(blocks)


def demo(bril):
    for func in bril['functions']:
        trivial_dce_pass(func)


if __name__ == '__main__':
    program = json.load(sys.stdin)
    demo(program)
    json.dump(program, sys.stdout, indent=2, sort_keys=True)
