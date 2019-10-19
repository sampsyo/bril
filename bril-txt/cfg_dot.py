"""Form a basic-block-based control-flow graph for a Bril function and
emit a GraphViz file.
"""

from form_blocks import form_blocks, TERMINATORS
import json
import sys
from collections import OrderedDict


def fresh(seed, names):
    """Generate a new name that is not in `names` starting with `seed`.
    """
    i = 1
    while True:
        name = seed + str(i)
        if name not in names:
            return name
        i += 1


def block_map(blocks):
    """Given a sequence of basic blocks, which are lists of instructions,
    produce a `OrderedDict` mapping names to blocks.

    The name of the block comes from the label it starts with, if any.
    Anonymous blocks, which don't start with a label, get an
    automatically generated name. Blocks in the mapping have their
    labels removed.
    """
    by_name = OrderedDict()

    for block in blocks:
        # Generate a name for the block.
        if 'label' in block[0]:
            # The block has a label. Remove the label but use it for the
            # block's name.
            name = block[0]['label']
            block = block[1:]
        else:
            # Make up a new name for this anonymous block.
            name = fresh('b', by_name)

        # Add the block to the mapping.
        by_name[name] = block

    return by_name


def successors(instr):
    """Get the list of jump target labels for an instruction.

    Raises a ValueError if the instruction is not a terminator (jump,
    branch, or return).
    """
    if instr['op'] == 'jmp':
        return instr['args']  # Just one destination.
    elif instr['op'] == 'br':
        return instr['args'][1:]  # The first argument is the condition.
    elif instr['op'] == 'ret':
        return []  # No successors to an exit block.
    else:
        raise ValueError('{} is not a terminator'.format(instr['op']))


def cfg_dot(bril, verbose):
    """Generate a GraphViz "dot" file showing the control flow graph for
    a Bril program.

    In `verbose` mode, include the instructions in the vertices.
    """
    for func in bril['functions']:
        print('digraph {} {{'.format(func['name']))

        blocks = block_map(form_blocks(func['instrs']))

        # Insert terminators into blocks that don't have them.
        for i, block in enumerate(blocks.values()):
            if block[-1]['op'] not in TERMINATORS:
                if i == len(blocks) - 1:
                    # In the last block, return.
                    block.append({'op': 'ret', 'args': []})
                else:
                    # Otherwise, jump to the next block.
                    dest = list(blocks.keys())[i + 1]
                    block.append({'op': 'jmp', 'args': [dest]})

        # Add the vertices.
        for name, block in blocks.items():
            if verbose:
                import briltxt
                print(r'  {} [shape=box, xlabel="{}", label="{}\l"];'.format(
                    name,
                    name,
                    r'\l'.join(briltxt.instr_to_string(i) for i in block),
                ))
            else:
                print('  {};'.format(name))

        # Add the control-flow edges.
        for i, (name, block) in enumerate(blocks.items()):
            succ = successors(block[-1])
            for label in succ:
                print('  {} -> {};'.format(name, label))

        print('}')


if __name__ == '__main__':
    cfg_dot(json.load(sys.stdin), '-v' in sys.argv[1:])
