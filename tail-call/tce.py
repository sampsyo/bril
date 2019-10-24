import sys
import json
sys.path.append('/Users/chrisroman/CornellWork/Fall_2019/CS6120/bril/examples')
from form_blocks import form_blocks
from cfg import *
from util import flatten, var_args
from collections import deque

# Gets all variables used in the function. We assume every variable will be
# assigned to at some point, so we'll just look for those that are assigned to
def get_vars(func):
    all_vars = set()
    all_vars.add("retval")
    for instr in func["instrs"]:
        if "op" in instr and "dest" in instr:
            all_vars.add(instr["dest"])
    return all_vars

def gen_copies(blocks, block):
    res = set()
    for instr in blocks[block]:
        if "op" in instr and instr["op"] == "id":
            res.add((instr["dest"], instr["args"][0]))
        elif "op" in instr and instr["op"] == "retval":
            res.add((instr["dest"], "retval"))
    return res

def kill_copies(blocks, block, all_vars):
    res = set()
    for instr in blocks[block]:
        if "op" in instr and instr["op"] != "id" and instr["op"] != "retval" \
           and "dest" in instr:
            # Add instr["dest"] X all_vars
            for copy in [(instr["dest"], v) for v in all_vars]:
                res.add(copy)
        if "op" in instr and instr["op"] == "call":
            # Add all_vars X "retval". After a call, nothign can be a copy
            # of retval
            for copy in [(v, "retval") for v in all_vars]:
                res.add(copy)

    return res

def start_block(preds):
    for (name, blocks) in preds.items():
        if blocks == []:
            return name

def copy_prop_func(func):
    all_vars = get_vars(func)
    top = set([(v1, v2) for v2 in all_vars for v1 in all_vars])
    blocks = block_map(form_blocks(func["instrs"]))
    add_terminators(blocks)
    preds, succs = edges(blocks)
    start = start_block(preds)
    ins = dict()
    outs = dict()

    print("initializing")
    # Initialize ins and outs
    for name in blocks:
        ins[name] = top.copy()
        outs[name] = set()
    ins[start] = set()
    outs[start] = gen_copies(blocks, start)

    # FIFO queue
    worklist = deque()
    print(start)
    worklist.append(start)
    while len(worklist) != 0:
        print(worklist)
        block_name = worklist.popleft()

        # Compute intersection over outs of preds
        acc = set() if not preds[block_name] else top.copy()
        for pred_name in preds[block_name]:
            acc = acc.intersection(outs[pred_name])
        ins[block_name] = acc

        # Compute new value of out
        old_out = outs[block_name].copy()
        outs[block_name] = \
            gen_copies(blocks, block_name).union(
                ins[block_name].difference(
                    kill_copies(blocks, block_name, all_vars)
                )
            )

        # If out value changed, add all successors to worklist
        print(old_out)
        if old_out != outs[block_name]:
            worklist.extend(succs[block_name])

    def find_copy(copies, rhs):
        for (src, dest) in copies:
            if dest == rhs:
                print(f"Replacing {rhs} with {src}")
                return src
        return None

    # Replace assignments with copies
    found_copy = False
    for (name, block) in blocks.items():
        for i, instr in enumerate(block):
            if "op" in instr and instr["op"] == "id":
                rhs_copy = find_copy(ins[name], instr["args"][0])
                if rhs_copy:
                    blocks[name][i]["args"][0] = rhs_copy
                    found_copy = True

    # TODO: Return whether or not to rerun copy_prop_func. Need to rerun
    # if any instruction was changed
    return found_copy

# Copy prop adapted from http://www.csd.uwo.ca/~moreno/CS447/Lectures/CodeOptimization.html/node8.html
def copy_prop(bril):
    for func in bril['functions']:
        found_copy = True
        while found_copy:
            found_copy = copy_prop_func(func)

def is_ret_no_args(instr):
    return "op" in instr and instr["op"] == "ret" and instr["args"] == []

def simple_tce_func(func):
    new_instrs = []
    instrs = func["instrs"]
    i = 0;
    while i < len(instrs):
        instr = instrs[i]
        # To do a tail call elimination, a call must be follow by:
        # - optionally, r: type = retval, and
        # - ret r or ret;
        # e.g.
        # ...
        # call foo
        # r: int = retval
        # ret r;
        if 'op' not in instr:
            new_instrs.append(instr)
            i += 1
        elif instr['op'] != 'call':
            new_instrs.append(instr)
            i += 1
        else:
            if i+1 < len(instrs) and is_ret_no_args(instrs[i+1]):
                new_instrs.append({"op": "jmp", "args": instr["args"]})
                i += 2
            elif i+2 < len(instrs) \
                 and "op" in instrs[i+1] and instrs[i+1]["op"] == 'retval' \
                 and "op" in instrs[i+2] and instrs[i+2]["op"] == 'ret' \
                 and (instrs[i+2]["args"] == [] or
                      instrs[i+2]["args"][0] == instrs[i+1]["dest"]):
                new_instrs.append({"op": "jmp", "args": instr["args"]})
                i += 3
            else:
                new_instrs.append(instr)
                i += 1
    func["instrs"] = new_instrs

def complex_tce(bril):
    blocks = block_map(form_blocks(func["instrs"]))
    add_terminators(blocks)
    preds, succs = edges(blocks)
    start = start_block(preds)

if __name__ == '__main__':
    bril = json.load(sys.stdin)
    for func in bril['functions']:
        simple_tce_func(func)

    #copy_prop(bril)
    #complex_tce(bril)

    json.dump(bril, sys.stdout, indent=2, sort_keys=True)
