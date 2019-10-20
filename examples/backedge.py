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

def findInitialIVValue(out_cprop,blocks, loop, iv):
    preds, succs = edges(blocks)
    cond_var_preds = preds[loop[0]]
    for pred in cond_var_preds:
        if (pred in loop):
            continue
        val = cprop_merge([out_cprop[pred]])
    if (val[iv] == '?'):
        return None
    else:
        return val[iv]

def findDelta(iv, br_cond_var_stmt_blockname, loop, out_rd, out_cprop):
    index_of_cond_var_block = loop.index(br_cond_var_stmt_blockname)
    index_of_block_before_cond_var_block = len(loop) - 1 if index_of_cond_var_block == 0 else index_of_cond_var_block - 1
    name_of_block_before_cond_var_block = loop[index_of_block_before_cond_var_block]

    delta_stmt = out_rd[name_of_block_before_cond_var_block][iv]
    assert(delta_stmt is not None)
    delta_var_name = next(a for a in delta_stmt["args"] if a != iv)
    delta_var_val = out_cprop[name_of_block_before_cond_var_block][delta_var_name]
    return delta_stmt["op"], delta_var_val

#Asserts strictly 1 branch stmt in entire loop
def findBranchStmt(blocks, loop):
    foundBranchStmt = None
    foundBranchBlockname = None
    for blockName in loop:
        for stmt in blocks[blockName]:
            if 'op' in stmt and stmt['op'] == 'br':
                assert(foundBranchStmt is None)
                foundBranchStmt = stmt
                foundBranchBlockname = blockName

    return foundBranchStmt, foundBranchBlockname

def findNameOfBlockWithStatement(blocks, targetStmt):
    for name, block in blocks.items():
        for stmt in block:
            if stmt is targetStmt:
                return name
    assert("Henry" == "Has Messed Up")

def findLoopInfo(bril, loops):
    blocks, in_cprop, out_cprop = run_df_return(bril, ANALYSES['cprop'])
    blocks, in_rd, out_rd = run_df_return(bril, ANALYSES['reachingDefs'])
    valid_op = ["eq", "lt", "gt", "ge", "le"]
    loop_infos = []

    for loop in loops:
        #Find branch statement
        branch_stmt, branch_blockname = findBranchStmt(blocks, loop)
        assert branch_stmt['op'] == 'br'
        
        #Find branch condition var
        br_cond_var = branch_stmt["args"][0]
        br_cond_var_stmt = out_rd[branch_blockname][br_cond_var]
        br_cond_var_stmt_blockname = findNameOfBlockWithStatement(blocks, br_cond_var_stmt)
        br_cond_var_stmt_op = br_cond_var_stmt['op']
        assert(br_cond_var_stmt_op in valid_op)

        #Find IV and bound
        iv_and_bound = br_cond_var_stmt['args']
        l = iv_and_bound[0]
        r = iv_and_bound[1]
        l_cprop = in_cprop[br_cond_var_stmt_blockname][l]
        r_cprop = in_cprop[br_cond_var_stmt_blockname][r]
        assert((l_cprop != '?') != (r_cprop != '?'))
        #Determine which is which
        bound_var = l if l_cprop != '?' else r
        iv = l if r_cprop != '?' else r
        #Get IV initial value and bound constant value
        bound_val = in_cprop[br_cond_var_stmt_blockname][bound_var]
        iv_val = findInitialIVValue(out_cprop,blocks, loop, iv)
        assert(iv_val is not None)

        #Find delta op and delta val
        delta_op, delta_var_val= findDelta(iv, br_cond_var_stmt_blockname, loop, out_rd, out_cprop)

        #Find base pointers
        base_pointers = findBasePointers(blocks, loop, in_cprop, iv, in_rd)
        
        loop_infos.append((loop, {'iv':iv, 'iv_val':iv_val, 'bound_var': bound_var, 'bound_val': bound_val, 'cond_op': br_cond_var_stmt_op, 'delta_op': delta_op, 'delta_val': delta_var_val, 'base_pointers': base_pointers}))

    return loop_infos

def filterEligibleLoops(loop_infos):
    out = []
    for LI in loop_infos:
        if LI[1]['delta_val'] != 1:
            continue
        out.append(LI)
    return out

def findBasePointers(blocks, loop, in_cprop, iv, in_rd):
    loadstores = []
    for blockName in loop:
        for stmt in blocks[blockName]:
            if 'op' in stmt and stmt['op'] == 'lw' or stmt['op'] == 'sw':
                loadstores.append(stmt)
    loadstore_pointers = [stmt['args'][-1] for stmt in loadstores]
    loadstore_bases = []
    loadstore_base_vals = []
    for index, pointer in enumerate(loadstore_pointers):
        loadstore_blockname = findNameOfBlockWithStatement(blocks, loadstores[index])
        pointer_stmt = in_rd[loadstore_blockname][pointer]
        base_var = next(arg for arg in pointer_stmt['args'] if arg != iv)
        loadstore_bases.append(base_var)
        pointer_stmt_blockname = findNameOfBlockWithStatement(blocks, pointer_stmt)
        base_val = in_cprop[pointer_stmt_blockname][base_var]
        # assert(base_val != '?')
        loadstore_base_vals.append(base_val)
        


    return dict(zip(loadstore_bases,loadstore_base_vals))

if __name__ == '__main__':
    bril = json.load(sys.stdin)
    res, blocks = backedge(bril)
    loops = findLoops(res, blocks)
    # for item in loops[1]:
    #     for item1 in blocks[item]:
    #         print(item1)
    loop_infos= findLoopInfo(bril, loops)
    for loop, info in loop_infos:
        print("LOOP: ", loop)
        print("INFO: ", info, '\n')
    filtered_loopInfos = filterEligibleLoops(loop_infos)


