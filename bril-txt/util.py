import itertools
from form_blocks import TERMINATORS


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


def fresh(seed, names):
    """Generate a new name that is not in `names` starting with `seed`.
    """
    i = 1
    while True:
        name = seed + str(i)
        if name not in names:
            return name
        i += 1
