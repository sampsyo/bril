import sys
import json
from cfg import edges
from df import run_df_return, ANALYSES, cprop_merge
from util import flatten

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

    in_aggregate = []
    for pred in cond_var_preds:
        if (pred in loop):
            continue
        in_aggregate.append(pred)
    val = cprop_merge([out_cprop[pred] for pred in in_aggregate])
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
    return delta_stmt["op"], delta_var_val, delta_stmt

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
        bound_var_def = in_rd[br_cond_var_stmt_blockname][bound_var]
        bound_val = in_cprop[br_cond_var_stmt_blockname][bound_var]
        iv_val = findInitialIVValue(out_cprop,blocks, loop, iv)
        assert(iv_val is not None)

        #Find delta op and delta val
        delta_op, delta_var_val, delta_stmt = findDelta(iv, br_cond_var_stmt_blockname, loop, out_rd, out_cprop)

        #Find base pointers
        base_pointers = findBasePointers(blocks, loop, in_cprop, iv, in_rd)
        
        loop_infos.append((loop, {'iv':iv, 'iv_val':iv_val,'bound_var_def':bound_var_def,
            'bound_var': bound_var, 'bound_val': bound_val, 'cond_op': br_cond_var_stmt_op,
            'br_cond_var_stmt': br_cond_var_stmt, 'delta_stmt': delta_stmt, 'delta_op': delta_op,
            'delta_val': delta_var_val, 'base_pointers': base_pointers}))

    return loop_infos, in_cprop

def filterEligibleLoops(loop_infos, blocks, in_cprop):
    out = []
    for li in loop_infos:
        loop = li[0]
        info = li[1]
        if info['delta_val'] != 1:
            continue
        if blocks[loop[-1]][-1]['op'] != 'br':
            continue
        loaded_vars = []
        offset_vars = [] #in the form const + i
        dip = False
        for blockName in loop:
            inst = blocks[blockName][0]
            if inst['op'] == 'lw':
                loaded_vars.append(inst['dest'])
            elif inst['op'] == 'sw':
                data_var = inst['args'][0]
                if data_var not in loaded_vars and in_cprop[blockName][data_var] == '?' or data_var == info['iv']:
                    dip = True
                    break
                
            elif inst['op'] == 'print':
                if inst['args'][0] in loaded_vars:
                    dip = True
                    break
            elif len(inst['args']) == 2:
                if inst['args'][0] in loaded_vars and inst['args'][1] in loaded_vars:
                    loaded_vars.append(inst['dest'])
        if dip:
            continue
        
        out.append((loop,info))
    return out

def findBasePointers(blocks, loop, in_cprop, iv, in_rd):
    loadstores = []
    for blockName in loop:
        for stmt in blocks[blockName]:
            if 'op' in stmt and stmt['op'] == 'lw' or stmt['op'] == 'sw':
                loadstores.append(stmt)
    loadstore_pointers = [stmt['args'][-1] for stmt in loadstores]
    loadstore_pointer_stmts = [in_rd[findNameOfBlockWithStatement(blocks, stmt)][stmt['args'][-1]] for stmt in loadstores] #Messy
    loadstore_bases = []
    loadstore_base_vals = []
    for index, pointer in enumerate(loadstore_pointers):
        loadstore_blockname = findNameOfBlockWithStatement(blocks, loadstores[index])
        pointer_stmt = in_rd[loadstore_blockname][pointer]
        base_var = next(arg for arg in pointer_stmt['args'] if arg != iv)
        loadstore_bases.append(base_var)
        pointer_stmt_blockname = findNameOfBlockWithStatement(blocks, pointer_stmt)
        base_val = in_cprop[pointer_stmt_blockname][base_var]
        loadstore_base_vals.append(base_val)

    return {'loadstore_stmts': loadstores,
    'loadstore_bases': loadstore_bases,
    'loadstore_base_vals': loadstore_base_vals, 
    'loadstore_pointers':loadstore_pointers, 
    'loadstore_pointer_stmts': loadstore_pointer_stmts}

#Delete block and handle in/out edges. Doesn't work for br blocks
def annihilateBlock(blocks, targetBlock):
    preds, succs = edges(blocks)

    assert(len(preds[targetBlock])==1 and len(succs[targetBlock]) == 1)
    assert(blocks[targetBlock][-1]['op'] != 'br')
    if(len(preds[targetBlock])==1 ):
        pred_exit_instr = blocks[preds[targetBlock][0]][-1]
        succ_block_name = succs[targetBlock][0]
        for index, arg in enumerate(pred_exit_instr['args']):
            if arg == targetBlock:
                pred_exit_instr['args'][index] = succ_block_name
        blocks.pop(targetBlock)
        

def loopToOneBlock(blocks, loop):
    assert(blocks[loop[-1]][-1]['op'] == 'br') 
    br_stmt = blocks[loop[-1]][-1]
    while len(loop) > 2:
        blockName = loop.pop(1)
        blockStmt = blocks[blockName][0]
        blocks[loop[0]].insert(-1, blockStmt)
        annihilateBlock(blocks, blockName)
    blockName = loop.pop(1)
    blockStmt = blocks[blockName][0]
    blocks[loop[0]].insert(-1, blockStmt)
    blocks[loop[0]][-1] = br_stmt
    blocks.pop(blockName)

def numIterations(initial_val, delta_op, bound_val, cond_op):
    num = 0
    i = initial_val
    while True:
        i += 1 if delta_op == 'add' else -1
        num+=1
        if cond_op == 'eq':
            if i == bound_val: return num
        elif cond_op == 'lt':
            if i < bound_val: return num
        elif cond_op == 'gt':
            if i > bound_val: return num
        elif cond_op == 'ge':
            if i >= bound_val: return num
        elif cond_op == 'le':
            if i <= bound_val: return num

def insertBlockAfter(firstBlockName, newBlock, blocks):
    old_target = blocks[firstBlockName][-1]['args']
    

def stripMine(filtered_loopInfos, blocks):
    loops = []
    for li in filtered_loopInfos:
        loops.append(li[0])
    for i in range(len(loops)):
        loop_info = filtered_loopInfos[i][1]

        num_iterations = numIterations(loop_info['iv_val'], loop_info['delta_op'], loop_info['bound_val'], loop_info['cond_op'])

        # Makes loop into 1 big block
        loopToOneBlock(blocks, loops[i])
        loopBlock = blocks[loops[i][0]]

        # Duplicate loop and connect to bottom if (n mod 4 != 0)
        if num_iterations % 4 != 0:
            branch_targs = loopBlock[-1]['args']
            cond_var = loopBlock[-1]['args'][0]
            branch_exit_targ = branch_targs[0] if branch_targs[0] != loops[i][0] else branch_targs[1]
            new_block_name = loops[i][0] + "COPY"
            new_block_stmts = [stmt for stmt in loopBlock[:-1]]
            new_block_stmts.append({'op':'br', 'args':[cond_var, branch_exit_targ, new_block_name]})
            blocks[new_block_name] = new_block_stmts

        loaded_vars = set()
        serial_vars = []
        # Traverse loop, change loads to vload(i)
        #Assert single-block loops
        assert(len(loops[i]) == 1)
        index = 0
        while index < len(blocks[loops[i][0]]):
            current_block = blocks[loops[i][0]]
            current_insn = current_block[index]
            assert('op' in current_insn)
            if 'op' in current_insn and current_insn['op'] == 'lw':
                current_insn["op"] = 'vload'
                loaded_vars.add(current_insn["dest"])
            elif current_insn is loop_info['delta_stmt']:
                index+=1
            # New Bound Variable Value
            elif current_insn is loop_info['bound_var_def']:
                #not necessarily in the loop...
                n = loop_info['bound_val']
                n_mod_four = n - (n % 4)
                blocks[block_name][0] = {'dest': loop_info["bound_var"], 'op': 'const', 'type': 'int', 'value': n_mod_four}
            elif 'op' in current_insn and current_insn['op'] == 'sw':
                current_insn["op"] = 'vstore'
            elif 'op' in current_insn and current_insn['op'] == 'add':
                if current_insn['args'][0] in loaded_vars and current_insn['args'][1] in loaded_vars:
                    current_insn["op"] = 'vadd'
                else:
                    serial_vars.append(current_insn)
            elif 'op' in current_insn and current_insn['op'] == 'sub':
                if current_insn['args'][0] in loaded_vars and current_insn['args'][1] in loaded_vars:
                    current_insn["op"] = 'vsub'
                else:
                    serial_vars.append(current_insn)
            elif 'op' in current_insn and current_insn['op'] == 'mul':
                if current_insn['args'][0] in loaded_vars and current_insn['args'][1] in loaded_vars:
                    current_insn["op"] = 'vmul'
                else:
                    serial_vars.append(current_insn)
            elif 'op' in current_insn and current_insn['op'] == 'div':
                if current_insn['args'][0] in loaded_vars and current_insn['args'][1] in loaded_vars: 
                    current_insn["op"] = 'vdiv'
                else:
                    serial_vars.append(current_insn)
            index +=1

        delta_index = next(ind for ind in range(len(blocks[loops[i][0]])) if blocks[loops[i][0]][ind] is loop_info['delta_stmt'])
        

        loopBlock.insert(delta_index+1, loop_info['delta_stmt'])
        for inst in serial_vars:
            loopBlock.insert(delta_index + 1, inst)
        
        loopBlock.insert(delta_index+1, loop_info['delta_stmt'])
        for inst in serial_vars:
            loopBlock.insert(delta_index + 1, inst)
        
        loopBlock.insert(delta_index+1, loop_info['delta_stmt'])
        for inst in serial_vars:
            loopBlock.insert(delta_index + 1, inst)

    return blocks

if __name__ == '__main__':
    bril = json.load(sys.stdin)
    res, blocks = backedge(bril)
    loops = findLoops(res, blocks)
    loop_infos, in_cprop = findLoopInfo(bril, loops)
    filtered_loopInfos = filterEligibleLoops(loop_infos, blocks, in_cprop)
    
    blocks = stripMine(filtered_loopInfos, blocks)

    for func in bril['functions']:
        func['instrs'] = []
        for name, block in blocks.items():
            func['instrs'].append({"label":name})
            func['instrs'].extend(block)
    json.dump(bril, sys.stdout, indent=2, sort_keys=True)

