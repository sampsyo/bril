"""Local value numbering for Bril.
"""
import json
import sys
from collections import namedtuple

from form_blocks import form_blocks, TERMINATORS
from util import flatten, var_args

# A Value uniquely represents a computation in terms of sub-values.
Value = namedtuple('Value', ['op', 'args'])


class Numbering(dict):
    """A dict mapping anything to numbers that can generate new numbers
    for you when adding new values.
    """
    def __init__(self, init={}):
        super(Numbering, self).__init__(init)
        self._next_fresh = 0

    def _fresh(self):
        n = self._next_fresh
        self._next_fresh = n + 1
        return n

    def add(self, key):
        """Associate the key with a new, fresh number and return it. The
        value may already be in the map; if so, it is overwritten and
        the old number is forgotten.
        """
        n = self._fresh()
        self[key] = n
        return n


def last_writes(instrs):
    """Given a block of instructions, return a list of bools---one per
    instruction---that indicates whether that instruction is the last
    write for its variable.
    """
    out = [False] * len(instrs)
    seen = set()
    for idx, instr in reversed(list(enumerate(instrs))):
        if 'dest' in instr:
            dest = instr['dest']
            if dest not in seen:
                out[idx] = True
                seen.add(instr['dest'])
    return out


def read_first(instrs):
    """Given a block of instructions, return a set of variable names
    that are read before they are written.
    """
    read = set()
    written = set()
    for instr in instrs:
        read.update(set(var_args(instr)) - written)
        if 'dest' in instr:
            written.add(instr['dest'])
    return read


def lvn_block(block):
    """Use local value numbering to optimize a basic block. Modify the
    instructions in place.
    """
    # The current value of every defined variable. We'll update this
    # every time a variable is modified. Different variables can have
    # the same value number (if they represent identical computations).
    var2num = Numbering()

    # The canonical variable holding a given value. Every time we're
    # forced to compute a new value, we'll keep track of it here so we
    # can reuse it later.
    value2num = {}

    # The *canonical* variable name holding a given numbered value.
    # There is only one canonical variable per value number (so this is
    # not the inverse of var2num).
    num2var = {}

    # Initialize the table with numbers for input variables. These
    # variables are their own canonical source.
    for var in read_first(block):
        num = var2num.add(var)
        num2var[num] = var

    for instr, last_write in zip(block, last_writes(block)):
        # Look up the value numbers for all variable arguments,
        # generating new numbers for unseen variables.
        argvars = var_args(instr)
        argnums = tuple(var2num[var] for var in argvars)

        # Value operations are candidates for replacement.
        val = None
        if 'dest' in instr and 'args' in instr:
            # Construct a Value for this computation.
            val = Value(instr['op'], argnums)

            # Is this value already stored in a variable?
            if val in value2num:
                # The value already exists; we can reuse it. Provide the
                # original value number to any subsequent uses.
                num = value2num[val]
                var2num[instr['dest']] = num

                # Replace this instruction with a copy.
                instr.update({
                    'op': 'id',
                    'args': [num2var[num]],
                })
                continue

        # If this instruction produces a result, give it a number.
        if 'dest' in instr:
            newnum = var2num.add(instr['dest'])

            # If this instruction computes a value, record the new
            # variable as its canonical source.
            if val:
                value2num[val] = newnum

            if last_write:
                # Preserve the variable name for other blocks.
                var = instr['dest']
            else:
                # We must put the value in a new variable so it can be
                # reused by another computation in the feature (in case
                # the current variable name is reassigned before then).
                var = 'lvn.{}'.format(newnum)

            # Record the variable name and update the instruction.
            num2var[newnum] = var
            instr['dest'] = var

        # Update argument variable names.
        if 'args' in instr:
            # Numbers not in the table are inputs and left unchanged.
            new_args = [num2var.get(n, v)
                        for n, v in zip(argnums, argvars)]
            if instr['op'] not in TERMINATORS:
                instr['args'] = new_args
            elif instr['op'] == 'br':
                instr['args'] += instr['args'][1:] + new_args


def lvn(bril):
    """Apply the local value numbering optimization to every basic block
    in every function.
    """
    for func in bril['functions']:
        blocks = list(form_blocks(func['instrs']))
        for block in blocks:
            lvn_block(block)
        func['instrs'] = flatten(blocks)


if __name__ == '__main__':
    bril = json.load(sys.stdin)
    lvn(bril)
    json.dump(bril, sys.stdout, indent=2, sort_keys=True)
