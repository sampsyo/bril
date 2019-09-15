import sys
import json
from form_blocks import form_blocks
import cfg


def df(bril):
    for func in bril['functions']:
        # Form the CFG.
        blocks = cfg.block_map(form_blocks(func['instrs']))
        cfg.add_terminators(blocks)
        preds, succs = cfg.edges(blocks)

        print(preds)
        print(succs)


if __name__ == '__main__':
    bril = json.load(sys.stdin)
    df(bril)
    json.dump(bril, sys.stdout, indent=2, sort_keys=True)
