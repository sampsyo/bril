"""Create and print out the basic blocks in a Bril function.
"""

import json
import sys

# Instructions that terminate a basic block.
TERMINATORS = 'br', 'jmp', 'ret'


def form_blocks(instrs):
    """Given a list of Bril instructions, generate a sequence of
    instruction lists representing the basic blocks in the program.

    Every instruction in `instr` will show up in exactly one block. Jump
    and branch instructions may only appear at the end of a block, and
    control can transfer only to the top of a basic block---so labels
    can only appear at the *start* of a basic block. Basic blocks may
    not be empty.
    """

    # Start with an empty block.
    cur_block = []

    for instr in instrs:
        if 'op' in instr:  # It's an instruction.
            # Add the instruction to the currently-being-formed block.
            cur_block.append(instr)

            # If this is a terminator (branching instruction), it's the
            # last instruction in the block. Finish this block and
            # start a new one.
            if instr['op'] in TERMINATORS:
                yield cur_block
                cur_block = []

        else:  # It's a label.
            # End the block here (if it contains anything).
            if cur_block:
                yield cur_block

            # Start a new block with the label.
            cur_block = [instr]

    # Produce the final block, if any.
    if cur_block:
        yield cur_block


def print_blocks(bril):
    """Given a Bril program, print out its basic blocks.
    """
    import briltxt

    func = bril['functions'][0]  # We only process one function.
    for block in form_blocks(func['instrs']):
        # Mark the block.
        leader = block[0]
        if 'label' in leader:
            print('block "{}":'.format(leader['label']))
            block = block[1:]  # Hide the label, for concision.
        else:
            print('anonymous block:')

        # Print the instructions.
        for instr in block:
            print('  {}'.format(briltxt.instr_to_string(instr)))


if __name__ == '__main__':
    print_blocks(json.load(sys.stdin))
