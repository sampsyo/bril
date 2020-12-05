import json
import sys

from cfg import block_map, add_terminators, add_entry, reassemble
from form_blocks import form_blocks


def func_from_ssa(func):
    blocks = block_map(form_blocks(func['instrs']))
    add_entry(blocks)
    add_terminators(blocks)

    # Replace each phi-node.
    for block in blocks.values():
        # Insert copies for each phi.
        for instr in block:
            if instr.get('op') == 'phi':
                dest = instr['dest']
                type = instr['type']
                for i, label in enumerate(instr['labels']):
                    var = instr['args'][i]

                    # Insert a copy in the predecessor block, before the
                    # terminator.
                    pred = blocks[label]
                    pred.insert(-1, {
                        'op': 'id',
                        'type': type,
                        'args': [var],
                        'dest': dest,
                    })

        # Remove all phis.
        new_block = [i for i in block if i.get('op') != 'phi']
        block[:] = new_block

    func['instrs'] = reassemble(blocks)


def from_ssa(bril):
    for func in bril['functions']:
        func_from_ssa(func)
    return bril


if __name__ == '__main__':
    print(json.dumps(from_ssa(json.load(sys.stdin)), indent=2, sort_keys=True))
