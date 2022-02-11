import json
import sys
from cfg import make_blocks


def eliminate_unused_vars(instructions):
    used_variables = set()
    for instr in instructions:
        if 'args' in instr:
            for var in instr['args']:
                used_variables.add(var)

    for i, instr in enumerate(instructions):
        if 'dest' in instr and instr['dest'] not in used_variables:
            print(f"Deleting {instr}", file=sys.stderr)
            instructions.remove(instr)

    return instructions


def eliminate_redundant_assigns_in_basic_block(block):
    assigned_but_unused = {}
    for i, instr in enumerate(block):
        if 'args' in instr:
            for arg in instr['args']:
                if arg in assigned_but_unused:
                    del assigned_but_unused[arg]
        if 'dest' in instr:
            if instr['dest'] in assigned_but_unused:
                del block[assigned_but_unused[instr['dest']]]
            assigned_but_unused[instr['dest']] = i


def do_dce():
    prog = json.load(sys.stdin)
    for func in prog['functions']:
        # Eliminate everything which just doesn't get used
        old_len = 0
        while len(func['instrs']) != old_len:
            old_len = len(func['instrs'])
            func['instrs'] = eliminate_unused_vars(func['instrs'])

        # Eliminate everything which gets reassigned before use
        blocks = make_blocks(func['instrs'])
        instructions = []
        for block in blocks:
            old_len = 0
            while len(block) != old_len:
                old_len = len(block)
                eliminate_redundant_assigns_in_basic_block(block)
            instructions.extend(block)
        func['instrs'] = instructions

    print(json.dumps(prog))


if __name__ == '__main__':
    do_dce()