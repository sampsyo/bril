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
def invloop(ins,outs,natloops,blocks):
    #ins, outs = df_worklist(blocks,ANALYSIS["rdefs"]) 


    # all reaching defs are outside of the loop
    # only one definition
    # definition is loop invariant
    for loop in natloops:
        for instr in loop:
            run_df(instr, analysis)
   


### natural blocks
def natblocks(bril):#list of blocks and names
    for func in bril['functions']:
        blocks = block_map(form_blocks(func['instrs']))
        add_terminators(blocks)
        ins, outs = df_worklist(blocks, ANALYSIS["rdefs"])
    return ins, outs, natloops


def reachers(blocks):
    rins, routs = df_worklist(blocks, ANALYSIS["rdefs"])

    return rins,routs

# detect LI instructions for a single natural loop
def invloop(blocks,rins,routs,natloop):
    boolmap = {}
    worklist = []
    for blockname in natloop:
        boolmap[blockname] = {False for i in range(len(blocks[blockname]))} 
        worklist.append(blockname)

    while len(worklist)>0:
        block = worklist.pop()
        
        
        for instr in block:
            
            
            boolmap= 0 
    # is it loop invariant 
    # does it dominate all uses
    # no other definitions of same variable
    # dominates all loop exits
    return boolmap

### move stuff
def codemot(bril):
    for func in bril['functions']:
        blocks = block_map(form_blocks(func['instrs']))
        add_terminators(blocks)
        rins, routs = reachers(blocks) # what is reaching
        for natloop in natloops(blocks):
            boolmap = invloop(blocks,rins,routs,natloop) 
            
    return new

def natloops(blocks): #input backedge
    pred,succ = edges(blocks)
    dom = get_dom(succ,list(blocks.keys())[0])
    for source,sink in backedge(succ,dom):
        yield loopsy(source,sink,pred) # natloops

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
