"""Local value numbering for Bril.
"""
import json
import sys

from form_blocks import form_blocks
from util import flatten


def lvn_block(block):
    return block


def lvn(bril):
    for func in bril['functions']:
        blocks = list(form_blocks(func['instrs']))
        new_blocks = [lvn_block(block) for block in blocks]
        func['instrs'] = flatten(new_blocks)


if __name__ == '__main__':
    bril = json.load(sys.stdin)
    lvn(bril)
    json.dump(bril, sys.stdout, indent=2, sort_keys=True)
