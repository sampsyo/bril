"""
Sameer Lal, sjl328.
CS 6120 Graduate Compilers.

Performs loop unrolling.
Baby: Natural loops (aiming to complete)
Adv: All types of loops (if time)
"""
from form_blocks import form_blocks, TERMINATORS
from cfg_dot import *
import cfg
import json
import sys
import dom
import util
from collections import OrderedDict

def cfg_edges(bril):
    """
        Generate list of edges in CFG.
    """
    edge_list = []
    for func in bril['functions']:
        blocks = block_map(form_blocks(func['instrs']))
        # Insert terminators into blocks that don't have them.
        for i, block in enumerate(blocks.values()):
            if block[-1]['op'] not in TERMINATORS:
                if i == len(blocks) - 1:
                    # In the last block, return.
                    block.append({'op': 'ret', 'args': []})
                else:
                    # Otherwise, jump to the next block.
                    dest = list(blocks.keys())[i + 1]
                    block.append({'op': 'jmp', 'args': [dest]})
        # Add the control-flow edges.
        for i, (name, block) in enumerate(blocks.items()):
            succ = successors(block[-1])
            for label in succ:
                edge_list.append((name, label))
        return edge_list


def find_backedges(cfg_edges, dom_dict):
    """
    Baby:  Returns a list of backedges in a CFG
    Adv: TODO: return only natural loops (i.e nonnested loops)
    """
    # Iterate through all edges (u,v) and see if v is dominated by u
    back_edges = []
    for e in cfg_edges:
        head, tail = e
        if tail in dom_dic[head]:
            back_edges.append(e)
    return back_edges

def loop_finder(bril, cfg_edges, dom_dic):
    """
    Looks at CFG and outputs blocks that form a loop
    In the first implementation we will naively look for cycles and 
    consider those to be loops.
    """
    loop_blocks = []
    back_edges = find_backedges(cfg_edges, dom_dic)

    #   Essentially we look for a backedge whose tail is dominated by the head
    #   and then we backtrack by populating a stack and predecessors
    #   to extract the loop body.
    for e in back_edges:
        n, h = e
        body = [h]
        stack = []
        stack.append(n)
        while( len(stack) ):
            d = stack.pop()
            if d not in body:
                body.append(d)
                # Search for predecessors
                # TODO optimize this using hash table in case of big graphs
                for ind, ele in enumerate(cfg_edges):
                    a, b = ele                    
                    if b == d:
                        stack.append(a)
        loop_blocks.append(body)
    return loop_blocks


def extract_variables_loop(bril, potential_loops, block_map):
    """
        Extract variables that are defined/modified in a loop excluding
        terminators.
    """
    potential_loops = potential_loops[0]
    all_variables = []
    # Obtain all variables
    for b in potential_loops:
        vb = []
        block = block_map[b]
        for inst in block:
            vb = vb + util.var_loop(inst)
        all_variables = all_variables +  vb
    all_variables = set(all_variables)

    return set(all_variables)

def is_switchable(bril, potential_loops, block_map):
    """
        Returns True/False indicating if the passed loops is switchable.
        optionally prints debug output indicating proof
    """

    # TODO: fix this so it returns an array per loop
    a = extract_variables_loop(bril, potential_loops, block_map)
    print(a)
    pass

def get_blocks_map(bril):
    for func in bril['functions']:
        blocks = block_map(form_blocks(func['instrs']))
    return blocks

if __name__ == '__main__':
    bril = json.load(sys.stdin)
    blocks_map = get_blocks_map(bril)
    edges = cfg_edges(bril)
    dom_dic = dom.get_dom_dict(bril)
    
    potential_loops = loop_finder(bril, edges, dom_dic)
    
    is_switchable(bril, potential_loops, blocks_map)


    

    
    
    