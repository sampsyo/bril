### project 2 Loop Detection

import json
import sys

from cfg import *
from form_blocks import *
from dom import *
from df import *


### detect back-edges
def get_backedges(successors,domtree):
    backedges = set()
    for source,sinks in successors.items():
        for sink in sinks:
            if sink in domtree[source]:
                backedges.add((source,sink))

    return backedges


### get the natural loop associated with an input backedge
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


### find all the natural loops
def natloops(blocks):
  pred,succ = edges(blocks)
  dom = get_dom(succ,list(blocks.keys())[0])
  for source,sink in get_backedges(succ,dom):
    yield loopsy(source,sink,pred)


### get all reaching definitions
def reachers(blocks):
    rins, routs = df_worklist(blocks, ANALYSIS["rdef"])
    return rins,routs


def reaching_def_vars(blocks, reaching_defs):
  rdef_vars = {}

  for blockname, rdefs_block in reaching_defs.items():
    block = blocks[blockname]
    block_rdef_vars = []

    for rdef_blockid, rdef_instr in rdefs_block:
      block_rdef_vars.append( \
          (rdef_blockid, rdef_instr, blocks[rdef_blockid][rdef_instr]["dest"]))

    rdef_vars[blockname] = block_rdef_vars

  return rdef_vars
    

### detect LI instructions for a single natural loop
def invloop(blocks,rdef_var_ins,rdef_var_outs,natloop):
    boolmap = {}
    worklist = []
    for blockname in natloop:
        boolmap[blockname] = [False for i in range(len(blocks[blockname]))]
        worklist.append(blockname)

    while len(worklist)>0:
      blockname = worklist.pop()
      block = blocks[blockname]

      boolmap_block = []
      for instr in block:
        # assignment of a constant to a variable
        if "dest" in instr and "value" in instr:
          boolmap_block.append(True)

        # assignment of a computation to a variable
        elif "dest" in instr and "args" in instr:
          # for each argument, either one of the following has to be true:
          # * all reaching defs of the argument are outside the loop
          # * there is exactly one reaching def for the argument in the loop
          instr_loop_invariant = True
          for arg in instr["args"]:
            var = instr["dest"]
            var_rdefs = list(filter(lambda rdef: rdef[2] == var, \
                rdef_var_ins[blockname]))
          
            var_rdefs_blocks = \
                map(lambda rdef: rdef[0] not in natloop, var_rdefs)

            rdefs_outside = all(var_rdefs_blocks)

            single_rdef = False
            if len(var_rdefs) == 1:
              rdef_block, rdef_instr, _ = var_rdefs[0]
              if rdef_block in natloop:
                single_rdef = boolmap[rdef_block][rdef_instr]

            instr_loop_invariant = \
                instr_loop_invariant and (rdefs_outside or single_rdef)

          boolmap_block.append(instr_loop_invariant)

        else:
          boolmap_block.append(False)
      
      boolmap[blockname] = boolmap_block


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
    rdef_var_ins  = reaching_def_vars(blocks, rins)
    rdef_var_outs = reaching_def_vars(blocks, rins)

    for natloop in natloops(blocks):
      boolmap = invloop(blocks,rdef_var_ins,rdef_var_outs,natloop) 

def printstuffs(bril):
  for func in bril['functions']:
    blocks = block_map(form_blocks(func['instrs']))
    add_terminators(blocks)
    pred,succ = edges(blocks)
    dom = get_dom(succ,list(blocks.keys())[0])
    backedges = get_backedges(succ,dom)
    for source,sink in get_backedges(succ,dom):
      natloops = loopsy(source,sink,pred)


if __name__ == '__main__':
    codemot(json.load(sys.stdin))


### eof
