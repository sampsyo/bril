"""Create and print out the basic blocks in a Bril function.
"""

import json
import sys

# Instructions that terminate a basic block.
TERMINATORS = 'br', 'jmp', 'ret'

nextFreshLabelNum = 1

def nextFreshLabel():
    global nextFreshLabelNum 
    nextFreshLabelNum += 1
    return ("____label" + str(nextFreshLabelNum))

def form_blocks(instrs, singletonBlocks = False):
    """Given a list of Bril instructions, generate a sequence of
    instruction lists representing the basic blocks in the program.

    Every instruction in `instr` will show up in exactly one block. Jump
    and branch instructions may only appear at the end of a block, and
    control can transfer only to the top of a basic block---so labels
    can only appear at the *start* of a basic block. Basic blocks may
    not be empty.
    """

    #Makes every statement a block.
    #A block is composed of 3 things:
    #1. A label
    #2. A non-control-flow statement
    #3. A control-flow instruction (br, jmp, or ret)
    #Pseudocode:
    # if block has more than 2 statements: dont()


    if singletonBlocks:
        i = 0
        while i < len(instrs)-1:
            true1 = 'op' in instrs[i] and instrs[i]['op'] not in TERMINATORS 
            true2 = 'op' in instrs[i+1] and instrs[i+1]['op'] not in TERMINATORS

            if true1 and true2:
                newLabel = nextFreshLabel()
                labelInst = {'label': newLabel}
                jumpInst = {'args': [newLabel], 'op': 'jmp'}
                instrs.insert(i+1, labelInst)
                instrs.insert(i+1, jumpInst)
            i+=1

    # for i in instrs:
    #     print(i)

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
