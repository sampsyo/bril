import collections
import json
import sys
sys.path.append('examples/')
sys.path.append('bril-txt/')
import briltxt
import cfg
import form_blocks

Edge = collections.namedtuple('Edge', ['s', 't'])


def reorder_blocks(function):
    blocks = cfg.block_map(form_blocks.form_blocks(function['instrs']))
    cfg.add_terminators(blocks)
    preds, succs = cfg.edges(blocks)
    # print(preds)
    # print(succs)
    return function


if __name__ == '__main__':
    bril_file = sys.argv[1]
    with open(bril_file) as f:
        program = json.loads(briltxt.parse_bril(f.read()))
    for function in program['functions']:
        blocks = cfg.block_map(form_blocks.form_blocks(function['instrs']))
        for instr in function['instrs']:
            if 'label' in instr:
                instr['block'] = instr['label']
            else:
                bs = [b for b in blocks if instr in blocks[b]]
                instr['block'] = [b for b in blocks if instr in blocks[b]][0]
    print(json.dumps(program, indent=2, sort_keys=True))
