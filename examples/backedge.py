import sys
import json
from cfg import edges
from df import run_df_return, ANALYSES, cprop_merge

def backedge(bril):
    blocks, in_, out = run_df_return(bril, ANALYSES['dominators'])
    preds, succs = edges(blocks)
    res = []

    for block in blocks:
        for succ in succs[block]:
            if (succ in out[block]):
                res.append((block, succ))
    return res, blocks
        
def findLoop(current, target, blocks, visited):
    if current == target:
        visited.append(current)
        return True, visited
    preds, succs = edges(blocks)
    myPreds = preds[current]
    for p in myPreds:
        visited.append(current)
        found, path = findLoop(p, target, blocks, visited)
        if found:
            return found, path
    return False, None


def findLoops(res, blocks):
    loops = []
    for r in res:
        _, path = findLoop(r[0], r[1], blocks, list())
        path.reverse()
        loops.append(path)
    return loops

def findInitialInductionVarialbeValue(out_cprop,blocks, header, last, iv):
    preds, succs = edges(blocks)
    preds = preds[header]
    for pred in preds:
        if (pred == last):
            continue
        val = cprop_merge([out_cprop[pred]])
    if (val[iv] == '?'):
        return None
    else:
        return val[iv]

# def findCondDef(loop, br_cond)
#     #iterate over loop and find the shit

def findDelta(iv, out_header_rd, out_last_block_rd, out_last_block_cprop):
    delta = out_last_block_rd[iv]
    var = next(a for a in delta["args"] if a != iv)
    var_val = out_last_block_cprop[var]
    return delta["op"], var_val

def findBranchStmt(blocks, loop):
    foundBranch = None
    for blockName in loop:
        for stmt in blocks[blockName]:
            if 'op' in stmt and stmt['op'] == 'br':
                assert(foundBranch is not None)
                findBranch = stmt
    return foundBranch

def findLoopInfo(bril, loops):
    blocks, in_cprop, out_cprop = run_df_return(bril, ANALYSES['cprop'])
    blocks, in_rd, out_rd = run_df_return(bril, ANALYSES['reachingDefs'])
    valid_op = ["eq", "lt", "gt", "ge", "le"]

    for loop in loops:
        block = blocks[loop[0]]
        stmt = block[-1]
        out_header = out_rd[loop[0]]
        out_cprop_header = out_cprop[loop[0]]
        in_header = in_rd[loop[0]]
        out_last_block = out_rd[loop[-1]]
        assert stmt['op'] == 'br'
        # print(stmt)
        br_cond_var = stmt["args"][0]
        br_cond = out_rd[loop[0]]
        br_cond = br_cond[br_cond_var]
        # cond op
        op = br_cond['op']
        if br_cond is None or 'op' not in br_cond or op not in valid_op:
            continue 
        iv_or_bound = br_cond['args']
        l = iv_or_bound[0]
        r = iv_or_bound[1]
        # print("header name", loop[0])
        # print("left value:", l)
        # print("right value:", r)
        # print("reaching defs from header for l", in_header[l])
        # print("reaching defs from header for r", in_header[r])
        l_const = in_header[l] is not None and in_header[l]['op'] == "const" and in_header[r] is None
        r_const = in_header[r] is not None and in_header[r]["op"] == "const" and in_header[l] is None
        # print("lconst" , l_const)
        # print("rconst" , r_const)
        assert l_const != r_const
        boundvar = iv_or_bound[0] if l_const else iv_or_bound[1]
        iv = iv_or_bound[0] if r_const else iv_or_bound[1]
        delta_op, val= findDelta(iv, out_header, out_last_block, out_cprop[loop[-1]])
        bound_val = out_cprop_header[boundvar]
        iv_val = findInitialInductionVarialbeValue(out_cprop, blocks, loop[0], loop[-1], iv)
        return iv, iv_val, boundvar, bound_val, op, delta_op, val



if __name__ == '__main__':
    bril = json.load(sys.stdin)
    res, blocks = backedge(bril)
    loops = findLoops(res, blocks)
    print("loops@@@@@@", loops)
    for item in loops[0]:
        for item1 in blocks[item]:
            print(item1)
    print("finding loop info")
    a, b, c, d, e, f, g = findLoopInfo(bril, loops)
    print (a,b,c,d,e,f,g)

