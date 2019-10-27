import json
import copy
from collections import OrderedDict
import sys, os
pn = os.path.join(os.path.dirname(__file__), "examples/")
sys.path.insert(0, "/Users/Cindy/Nextcloud/Cornell/cs6120/bril/examples")
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
    The value a dictionary taking block name as key and 
    loop invariant as value.
    '''
    loop_invariants = {}
    for loop in loops.keys():
        loop_invariants[loop] = {}
        for block in loops[loop]:
            loop_invariants[loop][block] = []
            for instr in blocks[block]: 
                if 'dest' in instr.keys():
                    if instr['op']=='const':
                    # constant
                        loop_invariants[loop][block].append(instr)
                    else:
                    # the instr takes some variables ,
                    # all variables has reaching defintion outside the loop or
                    # all variables has one reaching definition that is L.I.
                        defs = True
                        for var in instr['args']:
                            rd = reach_def[block][var] #var is defined at rd block
                            c1 = all([x not in loops[loop] for x in rd ])
                            li = loop_invariants[loop].get(rd[0]) #None or li codes
                            li = [] if li is None else li
                            c2 = len(rd)==1 and any([var == i['dest'] for i in li])
                            defs = (c1 or c2) and defs
                        if defs:
                            loop_invariants[loop][block].append(instr)
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
                    pre_header.update({x:name for x in loops[edge]})
                    break
    return pre_header, new_blocks
    
def move_LI(blocks, pre_header, loop_invariants, loops, dom, live_var, exits):
    '''
    This function returns:
    blocks: It's a modification to the blocks - input the function. It move 
    quanlified loop invariants to the preheader blocks of loops.
    licd: It's a dictionary where key is back edge of each loop and value is 
    instruction get code motioned.
    '''
    licd = dict()
    b_names = list(blocks.keys())
    for back_edge in loop_invariants:
        licd[back_edge] = []
        #definitions inside the loop (1)
        defs = [ins.get('dest') for b in loops[back_edge]
                for ins in blocks[b] if ins.get('dest')]
        for b_name in loop_invariants[back_edge]:
            #predecessor block of pre-header (pre-header is empty block) (2)
            ind = b_names.index(pre_header[b_name]) - 1
            for instr in loop_invariants[back_edge][b_name]:
                #exit blocks of dest where dest is live out (3)
                edest = [e for e in exits[back_edge] if instr['dest'] 
                        in live_var[1][e]]
                # 1. there is only one definition of dest in the loop
                # 2. dest is not live out of pre-header
                # 3. dest's block dominates all loop exits where dest is live out
                if ( defs.count(instr['dest']) == 1 and ind >= 0 and 
                    instr['dest'] not in live_var[1][b_names[ind]] and 
                    all([b_name in dom[e] for e in edest])
                ):
                    blocks[b_name].remove(instr)
                    instr['comment'] = 'code motion'
                    blocks[pre_header[b_name]].append(instr)
                    licd[back_edge].append(instr)
    return blocks, licd
    
def blocks_to_func(blocks, func):
    '''
    This function returns:
    new_func: The same dictionary of instrs as func. Func is unmodified input, 
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
def loopReduce():
    bril = json.load(sys.stdin)
    
    for i, func in enumerate(bril['functions']):
        exits, live_var, dom, oblocks, loops, reach_def, _ = loop_king(func)
        loop_invariants = find_LI(oblocks, loops, reach_def)
        pre_header, new_blocks = create_preheaders(oblocks, loops)
        code_motion, licd = move_LI(new_blocks, pre_header, loop_invariants, loops, dom, live_var, exits)
        bril['functions'][i] = blocks_to_func(code_motion, func)
        #print(licd)
    print(json.dumps(bril)) 
#if __name__ == '__main__':
#    loopReduce()
   
   
   
   
