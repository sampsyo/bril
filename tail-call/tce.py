import sys
import json
#from util import flatten, var_args
#from form_blocks import form_blocks

def is_ret_no_args(instr):
    return "op" in instr and instr["op"] == "ret" and instr["args"] == []

def tce_func(func):
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

if __name__ == '__main__':
    bril = json.load(sys.stdin)
    for func in bril['functions']:
        tce_func(func)
    json.dump(bril, sys.stdout, indent=2, sort_keys=True)
