import json
import copy
from collections import OrderedDict
import sys
sys.path.insert(0, "examples/")
from cfg import block_map, successors, add_terminators
from form_blocks import form_blocks
from dom import get_dom, get_pred
from util import fresh
from df import df_worklist, ANALYSES

def union(dicts):
    out = {}
    for s in dicts:
        for k,v in s.items():
            if k not in out:
                out.update({k:v})
            else:
                out[k] = list(set(out[k])|set(v))
    return out

def gen(blocks,node):

    out={}
    for i in blocks[node]:
        if 'dest' in i:
            out.update({i['dest']:[node]})
    return out

def transfer(blocks,node,inval):

    ''' Add new variable definitions to the invalue
    Also, rewrite common definitions using update function
    '''
    out =  inval.copy()
    out.update(gen(blocks,node))
    return out

def reach_defs (blocks, succs, preds):

    #forward approach for reaching defs problem

    first_block = list(blocks.keys())[0]
    in_edges = preds
    out_edges = succs

    in_ = {first_block:{}}
    out = {node:{} for node in blocks}

    worklist = list(blocks.keys())
    while worklist:
        node = worklist.pop(0)

        list_inp = []
        for n in in_edges[node]:
            list_inp.append(out[n])
        inval = union(list_inp)
        in_[node] = inval

        outval = transfer(blocks,node, inval)
        if outval != out[node]:
            out[node] = outval
            worklist += out_edges[node]

    return in_,out



def get_backedges (succ_list,dom):
    back_edges = []
    for name,successor in succ_list.items():
        for dominator in dom[name]:
            if dominator in successor:
                back_edges.append([name,dominator])
    return back_edges


def back_search(node,explored,pred,loop):

    if node in explored:
        return
    explored.add(node)

    for pre in pred[node]:
        back_search(pre,explored,pred,loop)

    loop.append(node)

def find_loops(back_edges,pred):

    natural_loop = {}
    for t,h in back_edges:
        loop = []
        #find all blocks which reach t without h
        explored = set()
        explored.add(h) #header already explored
        loop.append(h) #adding header to loop

        back_search(t,explored,pred,loop)
        natural_loop[(t,h)] = loop

    return natural_loop

def find_exits(succ, loops):
    exits = {}
    for k in loops:
        exits[k] = []
        for l in loops[k]: 
            if any( s not in loops[k] for s in succ[l]):
                exits[k].append(l)
    return exits

def loop_king(func):
    ''' 
    This function returns:
    exits: A dictionary of exits of loops. 
    The key is the tuple of (tail,header) where the header dominates the tail and 
    there is an edge from tail -> header.
    The value is list of exit blocks.
    
    live_var: live variables.
    
    dom: A dictionary that maps each block with blocks that dominating it.
    
    original blocks: A list of blocks that non teminator is added. This is to 
    recover json from modified blocks
    
    back_edges: A list of pairs of [tail,header] where the header dominates the
    tail and there is an edge from tail -> header
    
    natural_loops: A dictionary of loops. 
    The key is the tuple of (tail,header) where the header dominates the tail and 
    there is an edge from tail -> header.
    The value is the list of lists of blocks. Each element of the list
    represents blocks present in a natural loop corresponding to the backedge in
    the previous list. So natural_loops[i] is the list of blocks in the loop
    represented by the backedge[i] edge.

    reaching_in and reaching out: These contain a dictionary of dictionary. Each
    key represents a block. For each block we have a dictionary of {variables:
    [reaching def block numbers]}. So to check if a variable has all the
    reaching definition before the loop, just check the variable in the reaching
    in def and see if the block number(s) listed is not in natural_loops list
    for that loop.
    '''
    #preprocessing
    blocks = block_map(form_blocks(func['instrs']))
    original_blocks = copy.deepcopy(blocks)
    add_terminators(blocks)
    
    succ = {name: successors(block[-1]) for name, block in blocks.items()}
    pred = get_pred(succ)
    dom = get_dom(succ, list(blocks.keys())[0])
    back_edges = get_backedges(succ,dom) 
    natural_loops = find_loops(back_edges,pred)
    reaching_in,reaching_out = reach_defs(blocks, succ, pred)
    
    exits = find_exits(succ, natural_loops)
    live_var = df_worklist(blocks,ANALYSES['live'])
    return exits, live_var, dom, original_blocks, natural_loops, reaching_in, reaching_out

def find_LI(blocks, loops, reach_def):
    ''' 
    This function returns:
    loop_invariants: a dictionary of loop invaraient. 
    The key is tuple(tail, head) of a loop backedge. 
    The value is loop invariants inside the loop.
    '''
    loop_invariants = {}
    for loop in loops.keys():
        loop_invariants[loop] = []  
        for block in loops[loop]:
            for instr in blocks[block]: 
                if 'dest' in instr.keys():
                    if instr['op']=='const':
                    # constant
                        loop_invariants[loop].append(instr['dest'])
                    else:
                    # the instr takes some variables ,
                    # all variables has reaching defintion outside the loop
                    # or has one reaching definition that is L.I.
                        var = instr['args']
                        defs = [ (y not in loops[loop] for y in reach_def[block][x])
                                or (x in loop_invariants[loop] 
                                and len(reach_def[block][x]) == 1)
                                for x in var ]
                        if all(defs):
                            loop_invariants[loop].append(instr['dest'])
    return loop_invariants


    
def create_preheaders(blocks, loops):
    '''
    This function returns:
    pre_header: A dictionary of blocks matching to its prehdeader block.
    
    new_blocks: Compared to the blocks as input of the function,
    the new_blocks have empty block at the predecessor block of 
    loop header block.
    '''
    new_blocks = OrderedDict()
    b_names = list(blocks.keys())
    pre_header = {}
    for i, k in enumerate(b_names):
        new_blocks[k] = blocks[k]
        if i+1 < len(b_names):
            for edge in loops:
                if b_names[i+1] in edge[1]:
                    name = fresh('b', new_blocks)
                    new_blocks[name] = []
                    pre_header = {x:name for x in loops[edge]}
                    break
    return pre_header, new_blocks
    
def move_LI(blocks, pre_header, loop_invariants, loops, dom, live_var, exits):
    '''
    This function returns:
    blocks: It's a modification to the blocks - input the function. It move 
    quantlified loop invariants to the preheader blocks of loops.
    '''
    b_names = list(blocks.keys())
    for back_edge in loops:
        for b_name in loops[back_edge]:
            for instr in blocks[b_name]:
                if 'dest' in instr and instr['dest'] in loop_invariants[back_edge]:
                    #exist block of dest where dest is live out 
                    edest = [e for e in exits[back_edge] 
                            if instr['dest'] in live_var[1][e]]
                    #predecessor block of pre header
                    ind = b_names.index(pre_header[b_name])- 1 
                    # 1. there is only one definition of dest in the loop
                    # 2. dest's block dominates all loop exists where dest is live out
                    # 3. dest is not live out of pre-header
                    if ( loop_invariants[back_edge].count(instr['dest'])==1 and
                         all([b_name in dom[e]  for e in edest]) and
                         ind >= 0 and instr['dest'] not in live_var[1][b_names[ind]]
                        ):
                        blocks[b_name].remove(instr)
                        blocks[pre_header[b_name]].append(instr)
    return blocks
    
def blocks_to_func(blocks, func):
    '''
    This function returns:
    new_func: The same ordered dictionary type as func. Func is unmodified input, 
    while new_func is the modified version. It is generated according to blocks. 
    Because blocks has some label block (pre headers) that func does not have, 
    and these blocks has sequential execution order, it's fine to just remove 
    the labels.
    This is because we want to make the program unchanged except for loop invariant 
    code motion and strength reduction. Newly added labels shouldn't be in 
    the program.
    '''
    new_instrs = []
    label_name = []
    for instr in func['instrs']:
        if 'label' in instr: label_name.append(instr['label'])
    for block in blocks:
        if block in label_name:
            new_instrs.append({'label':block})
        new_instrs = new_instrs+blocks[block]
    new_func = copy.deepcopy(func)
    new_func['instrs'] = new_instrs
    return new_func
    
if __name__ == '__main__':
    bril = json.load(sys.stdin)
    for func in bril['functions']:
        exits, live_var, dom, oblocks, loops, reach_def, _ = loop_king(func)
        loop_invariants = find_LI(oblocks, loops, reach_def)
        pre_header, new_blocks = create_preheaders(oblocks, loops)
        code_motion = move_LI(new_blocks, pre_header, loop_invariants, loops, dom, live_var, exits)
        blocks_to_func(code_motion, func)
   
   
   
   
