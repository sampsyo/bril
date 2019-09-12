"""Local value numbering for Bril.
"""
import json
import sys
from collections import namedtuple

from form_blocks import form_blocks
from util import flatten

# A Value uniquely represents a computation in terms of sub-values.
Value = namedtuple('Value', ['op', 'args'])


# class Numbering(dict):
#     """A dict mapping anything numbers that automatically grows to
#     assign new numbers to unknown values when they're first looked up.
#     """
#     def __init__(self, init={}):
#         super(Numbering, self).__init__(init)
#         self._fresh = 0

#     def _get_fresh(self):
#         n = self._fresh
#         self._fresh = n + 1
#         return n

#     def __getitem__(self, key):
#         if key in self:
#             return key[self]
#         else:
#             return self.add(key)

#     def add(self, key):
#         assert key not in self
#         value = self._get_fresh()
#         self[key] = value
#         return value


def lvn_block(block):
    # The current value of every defined variable. We'll update this
    # every time a variable is modified. Initially, all variables get
    # distinct numbers, in case they're used as inputs without
    # modification.
    var2num = {}

    # The canonical variable holding a given value. Every time we're
    # forced to compute a new value, we'll keep track of it here so we
    # can reuse it later.
    value2num = {}

    num2var = {}

    fresh = 0

    def getnum(var):
        if var in var2num:
            return var2num[var]
        else:
            nonlocal fresh
            num = fresh
            fresh = fresh + 1
            var2num[var] = num
            num2var[num] = var
            return num

    for idx, instr in enumerate(block):
        # Only deal with value operations.
        if 'dest' in instr and 'args' in instr:
            # Construct a Value for this expression, using the numbers
            # for its arguments.
            argnums = tuple(getnum(var) for var in instr['args'])
            val = Value(instr['op'], argnums)

            # Is this variable already stored in a variable?
            if val in value2num:
                # The value already exists; we can reuse it. Provide the
                # original value number to any subsequent uses.
                num = value2num[val]
                var2num[instr['dest']] = num

                # Replace this instruction with a copy.
                yield {
                    'op': 'id',
                    'dest': instr['dest'],
                    'type': instr['type'],
                    'args': [num2var[num]],
                }

            else:
                # We actually need to compute something. Create a new
                # number for this value.
                newnum = getnum(instr['dest'])  # var2num.add(instr['dest'])
                value2num[val] = newnum

                # We must put the value in a new variable so it can be
                # reused by another computation in the feature (in case
                # the current variable name is reassigned before then).
                newvar = 'lvn.{}'.format(newnum)
                num2var[newnum] = newvar

                # Reconstruct the operation with new arguments.
                yield {
                    'op': instr['op'],
                    'dest': newvar,
                    'type': instr['type'],
                    'args': [num2var[n] for n in argnums],
                }

        else:
            yield instr


def lvn(bril):
    for func in bril['functions']:
        blocks = list(form_blocks(func['instrs']))
        new_blocks = [list(lvn_block(block)) for block in blocks]
        func['instrs'] = flatten(new_blocks)


if __name__ == '__main__':
    bril = json.load(sys.stdin)
    lvn(bril)
    json.dump(bril, sys.stdout, indent=2, sort_keys=True)
