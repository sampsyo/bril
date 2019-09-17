import sys
import json
from form_blocks import form_blocks
import cfg
from collections import namedtuple

# A single dataflow analysis consists of these part:
# - forward: True for forward, False for backward.
# - init: An initial value (bottom or top of the latice).
# - merge: Take a list of values and produce a single value.
# - transfer: The transfer function.
Analysis = namedtuple('Analysis', ['forward', 'init', 'merge', 'transfer'])


def df_worklist(blocks, analysis):
    preds, succs = cfg.edges(blocks)

    # Switch between directions.
    if analysis.forward:
        first_block = list(blocks.keys())[0]  # Entry.
        in_edges = preds
        out_edges = succs
    else:
        first_block = list(blocks.keys())[-1]  # Exit.
        in_edges = succs
        out_edges = preds

    # Initialize.
    in_ = {first_block: analysis.init}
    out = {node: analysis.init for node in blocks}

    # Iterate.
    worklist = list(blocks.keys())
    while worklist:
        node = worklist.pop(0)

        inval = analysis.merge(out[n] for n in in_edges[node])
        in_[node] = inval

        outval = analysis.transfer(blocks[node], inval)

        if outval != out[node]:
            out[node] = outval
            worklist += out_edges[node]

    if analysis.forward:
        return in_, out
    else:
        return out, in_


def gen(block):
    return {i['dest'] for i in block if 'dest' in i}


def kill(block):
    return {i['dest'] for i in block if 'dest' in i}


def union(sets):
    out = set()
    for s in sets:
        out.update(s)
    return out


ANALYSES = {
    'defined': Analysis(
        True,
        init=set(),
        merge=union,
        transfer=lambda block, in_: gen(block).union(in_ - kill(block)),
    )
}


def run_df(bril):
    for func in bril['functions']:
        # Form the CFG.
        blocks = cfg.block_map(form_blocks(func['instrs']))
        cfg.add_terminators(blocks)

        in_, out = df_worklist(blocks, ANALYSES['defined'])
        for block in blocks:
            print('{}:', block)
            print('  in: ', in_[block])
            print('  out:', out[block])


if __name__ == '__main__':
    bril = json.load(sys.stdin)
    run_df(bril)
