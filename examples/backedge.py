import sys
import json
from cfg import edges
from df import run_df_return, ANALYSES

def backedge(bril):
    blocks, in_, out = run_df_return(bril, ANALYSES['dominators'])
    preds, succs = edges(blocks)
    res = []

    for block in blocks:
        for succ in succs[block]:
            if (succ in out[block]):
                res.append((block, succ))
    return res
        

if __name__ == '__main__':
    bril = json.load(sys.stdin)
    res = backedge(bril)
    print (res)


