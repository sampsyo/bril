"""Form a basic-block-based control-flow graph for a Bril function and
emit a GraphViz file.
"""

from form_blocks import form_blocks
import json
import sys
from cfg import block_map, successors, add_terminators


def cfg_dot(bril, verbose):
    """Generate a GraphViz "dot" file showing the control flow graph for
    a Bril program.

    In `verbose` mode, include the instructions in the vertices.
    """
    for func in bril['functions']:
        print('digraph {} {{'.format(func['name']))

        blocks = block_map(form_blocks(func['instrs']))

        # Insert terminators into blocks that don't have them.
        add_terminators(blocks)

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
