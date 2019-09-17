import sys
import json
from form_blocks import form_blocks
import cfg


def df_worklist(blocks, bottom, merge, transfer):
    preds, succs = cfg.edges(blocks)
    entry = list(blocks.keys())[0]

    # Initialize.
    invals = {entry: bottom}
    outvals = {node: bottom for node in blocks}
    worklist = list(blocks.keys())

    # Iterate.
    while worklist:
        node = worklist.pop(0)

        inval = merge(outvals[n] for n in preds[node])
        invals[node] = inval

        outval = transfer(blocks[node], inval)

        if outval != outvals[node]:
            outvals[node] = outval
            worklist += succs[node]

    print(invals)
    print(outvals)


def gen(block):
    return {i['dest'] for i in block if 'dest' in i}


def kill(block):
    return {i['dest'] for i in block if 'dest' in i}


def _union(sets):
    out = set()
    for s in sets:
        out.update(s)
    return out


def run_df(bril):
    for func in bril['functions']:
        # Form the CFG.
        blocks = cfg.block_map(form_blocks(func['instrs']))
        cfg.add_terminators(blocks)

        df_worklist(
            blocks,
            bottom=set(),
            merge=_union,
            transfer=lambda block, in_: gen(block).union(in_ - kill(block)),
        )


if __name__ == '__main__':
    bril = json.load(sys.stdin)
    run_df(bril)
    json.dump(bril, sys.stdout, indent=2, sort_keys=True)
