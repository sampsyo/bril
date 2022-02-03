import json
import sys

TERMINATORS = {'jmp', 'br', 'ret'}


def make_blocks(body):
    cur_block = []
    for instr in body:
        if 'op' in instr:
            cur_block.append(instr)
            if instr['op'] in TERMINATORS:
                if cur_block: yield cur_block
                cur_block = []
        elif 'label' in instr:
            if cur_block: yield cur_block
            cur_block = [instr]
        else:
            assert False, f"{instr} has neither an op nor label"
    if cur_block: yield cur_block


def name_blocks(blocks):
    named_blocks = []
    for block in blocks:
        name = f"b{len(named_blocks)}"
        print(block)
        if 'label' in block[0]:
            name = block[0]['label']
        named_blocks.append((name, block))
    return named_blocks


def make_cfg():
    prog = json.load(sys.stdin)
    cfg = {}
    for func in prog['functions']:
        named_blocks = name_blocks(make_blocks(func['instrs']))
        for i, (name, block) in enumerate(named_blocks):
            if 'op' in block[-1] and block[-1]['op'] in {'jmp', 'br'}:
                cfg[name] = block[-1]['labels']
            elif 'op' in block[-1] and block[-1]['op'] == 'ret':
                cfg[name] = []
            else:
                if i + 1 < len(named_blocks):
                    cfg[name] = [named_blocks[i+1][0]]
                else:
                    cfg[name] = []
    print(cfg)


if __name__ == "__main__":
    make_cfg()
