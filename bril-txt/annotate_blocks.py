import collections
import json
import sys
sys.path.append('examples/')
import cfg
import form_blocks

Edge = collections.namedtuple('Edge', ['s', 't'])


def reorder_blocks(function):
    blocks = cfg.block_map(form_blocks.form_blocks(function['instrs']))
    cfg.add_terminators(blocks)
    preds, succs = cfg.edges(blocks)
    print(preds)
    print(succs)
    return function


if __name__ == '__main__':
    bril_file = sys.argv[1]
    with open(bril_file) as f:
        program = json.load(f)
    new_program = dict(functions=list())
    for function in program['functions']:
        new_program['functions'].append(reorder_blocks(function))
    print(new_program)
