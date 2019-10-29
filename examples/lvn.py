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


def lvn_block(block, lookup, canonicalize, fold):
    """Use local value numbering to optimize a basic block. Modify the
    instructions in place.

    You can extend the basic LVN algorithm to bring interesting language
    semantics with these functions:

    - `lookup`. Arguments: a value-to-number map and a value. Return the
      corresponding number (or None if it does not exist).
    - `canonicalize`. Argument: a value. Returns an equivalent value in
      a canonical form.
    - `fold`. Arguments: a number-to-constant map  and a value. Return a
      new constant if it can be computed directly (or None otherwise).
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

    # Track constant values for values assigned with `const`.
    num2const = {}

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
            val = canonicalize(Value(instr['op'], argnums))

            # Is this value already available?
            num = lookup(value2num, val)
            if num is not None:
                # Mark this variable as containing the value.
                var2num[instr['dest']] = num

                # Replace the instruction with a copy or a constant.
                if num in num2const:  # Value is a constant.
                    instr.update({
                        'op': 'const',
                        'value': num2const[num],
                    })
                    del instr['args']
                else:  # Value is in a variable.
                    instr.update({
                        'op': 'id',
                        'args': [num2var[num]],
                    })
                continue

        # If this instruction produces a result, give it a number.
        if 'dest' in instr:
            newnum = var2num.add(instr['dest'])

            # Record constant values.
            if instr['op'] == 'const':
                num2const[newnum] = instr['value']

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

            if val:
                # Is this value foldable to a constant?
                const = fold(num2const, val)
                if const:
                    num2const[newnum] = const
                    instr.update({
                        'op': 'const',
                        'value': const,
                    })
                    del instr['args']
                    continue

                # If not, record the new variable as the canonical
                # source for the newly computed value.
                value2num[val] = newnum

        # Update argument variable names to canonical variables.
        if 'args' in instr:
            new_args = [num2var[n] for n in argnums]
            if instr['op'] not in TERMINATORS:
                instr['args'] = new_args
            elif instr['op'] == 'br':
                instr['args'] += instr['args'][1:] + new_args


def _lookup(value2num, value):
    """Value lookup function with propagation through `id` values.
    """
    if value.op == 'id':
        return value.args[0]  # Use the underlying value number.
    else:
        return value2num.get(value)


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


def _fold(num2const, value):
    if value.op in FOLDABLE_OPS:
        try:
            const_args = [num2const[n] for n in value.args]
            return FOLDABLE_OPS[value.op](*const_args)
        except KeyError:  # At least one argument is not a constant.
            return None
        except ZeroDivisionError: # If we hit a dynamic error, bail!
            return None
    else:
        return None


def _canonicalize(value):
    """Cannibalize values for commutative math operators.
    """
    if value.op in ('add', 'mul'):
        return Value(value.op, tuple(sorted(value.args)))
    else:
        return value


def lvn(bril, prop=False, canon=False, fold=False):
    """Apply the local value numbering optimization to every basic block
    in every function.
    """
    for func in bril['functions']:
        blocks = list(form_blocks(func['instrs']))
        for block in blocks:
            lvn_block(
                block,
                lookup=_lookup if prop else lambda v2n, v: v2n.get(v),
                canonicalize=_canonicalize if canon else lambda v: v,
                fold=_fold if fold else lambda n2c, v: None,
            )
        func['instrs'] = flatten(blocks)


if __name__ == '__main__':
    bril = json.load(sys.stdin)
    lvn(bril, '-p' in sys.argv, '-c' in sys.argv, '-f' in sys.argv)
    json.dump(bril, sys.stdout, indent=2, sort_keys=True)
