### project 2 Loop Detection

import json
import sys

from cfg import *
from form_blocks import *
from dom import *
from df import *

### detect back-edges
def backedge(successors,domtree):
    backedges = set()
    for source,sinks in successors.items():
        for sink in sinks:
            if sink in domtree[source]:
                backedges.add((source,sink))

    return backedges

### get natural loops
def loopsy(source,sink,predecessors):
    worklist = [source]
    loop = set()
    while len(worklist)>0:
        current = worklist.pop()
        pr = predecessors[current]
        for p in pr:
            if not(p in loop or p==sink):
                loop.add(p)
                worklist.append(p)

    loop.add(sink)
    loop.add(source)
    return loop

### mark stuff as loop invariant
def invloop(ins,outs,natloops,blockmap):
    for loop in natloops:
        for instr in loop:
            run_df(instr, analysis)


### natural blocks
def natblocks(#list of blocks and names
    blocks = block_map(form_blocks(func_instrs))
    add_terminators(blocks)
    ins, outs = df_worklist(blocks, ANALYSIS["rdefs"])
    return ins, outs

### move stuff
def codemot(blocks):
    # is it loop invariant 
    # does it dominate all uses
    # no other definitions of same variable
    # dominates all loop exits
    # move block
    return new

###

def printstuffs(bril):
    for func in bril['functions']:
        blocks = block_map(form_blocks(func['instrs']))
        add_terminators(blocks)
        pred,succ = edges(blocks)
        dom = get_dom(succ,list(blocks.keys())[0])
        print("backedges: ",backedge(succ,dom))
        for source,sink in backedge(succ,dom):
            print("loops: ", loopsy(source,sink,pred))

if __name__ == '__main__':
    printstuffs(json.load(sys.stdin))


### eof
