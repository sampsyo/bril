import json
import sys

import sys
sys.path.insert(0, "examples/")

from cfg import block_map, successors, add_terminators
from form_blocks import form_blocks
from dom import get_dom, get_pred


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

    natural_loop = []
    for t,h in back_edges:
        loop = []
        #find all blocks which reach t without h
        explored = set()
        explored.add(h) #header already explored
        loop.append(h) #adding header to loop

        back_search(t,explored,pred,loop)
        natural_loop.append(loop)

    return natural_loop




def loop_king(bril):
    ''' This function returns:
    back_edges: A list of pairs of [tail,header] where the header dominates the
    tail and there is an edge from tail -> header
    
    natural_loops: A list of lists of blocks. Each element of the list
    represents blocks present in a natural loop corresponding to the backedge in
    the previous list. So natural_loops[i] is the list of blocks in the loop
    represented by the backedge[i] edge

    reaching_in and reaching out: These contain a dictionary of dictionary. Each
    key represents a block. For each block we have a dictionary of {variables:
    [reaching def block numbers]}. So to check if a variable has all the
    reaching definition before the loop, just check the variable in the reaching
    in def and see if the block number(s) listed is not in natural_loops list
    for that loop.
    '''
    for func in bril['functions']:
        blocks = block_map(form_blocks(func['instrs']))
        add_terminators(blocks)
        succ = {name: successors(block[-1]) for name, block in blocks.items()}
        pred = get_pred(succ)
        dom = get_dom(succ, list(blocks.keys())[0])

        back_edges = get_backedges (succ,dom) 
        natural_loops = find_loops(back_edges,pred)

        reaching_in,reaching_out = reach_defs (blocks, succ, pred)

        return back_edges, natural_loops, reaching_in, reaching_out

if __name__ == '__main__':
   loop_king(json.load(sys.stdin))
