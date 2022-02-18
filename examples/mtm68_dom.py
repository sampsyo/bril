import sys
import json

from form_blocks import form_blocks, block_name
from mtm68_cfg import Cfg

def intersection(doms, preds):
    if not preds:
        return set()
    else:
        for j, pred in enumerate(preds):
            name = block_name(pred)
            if j == 0:
                i = doms[name]
            else:
                i = i.intersection(doms[name])
        return i

def post_order(block, cfg, visited):
    order = []
    succs = cfg.get_succ(block_name(block))
    for succ in succs:
        if block_name(succ) not in visited:
            visited.append(block_name(block))
            order.extend(post_order(succ, cfg, visited))
    order.append(block)
    return order

def find_doms(func):
    blocks = list(form_blocks(func['instrs']))
    cfg = Cfg(blocks)
    reverse_post_order = list(reversed(post_order(blocks[0], cfg, [])))
    doms = {block_name(block) : set() for block in blocks}
    dom_changing = True
    while dom_changing:
        for block in reverse_post_order:
            name = block_name(block)
            preds = cfg.get_pred(name)
            len_before = 0 if name not in doms else len(doms[name])
            doms[name] = {name}.union(intersection(doms, preds))
            len_after = len(doms[name])
            dom_changing = len_before != len_after
    return doms

class Tree:
    def __init__(self, root, children = None):
        self.root = root
        self.children = children

def reachable(name, cfg, visited):
    r = set()
    succs = cfg.get_succ(name)
    for succ in succs:
        succ_name = block_name(succ)
        if succ_name not in visited:
            visited.add(succ_name)
            r = reachable(succ_name, cfg, visited)
            r.add(name)
    return r

def dom_tree_aux(name, doms):
    children = doms[name].remove(name)
    if not children:
        return Tree(name, None)
    else:
        ctrees = []
        for c in children:
            ctrees.append(dom_tree_aux(c, doms))
        return Tree(name, ctrees)

def dom_tree(func):
    blocks = list(form_blocks(func['instrs']))
    doms = find_doms(func)
    doms_inv = {block_name(block) : [] for block in blocks}
    for b, dominated_by_b in doms.items():
        for n in dominated_by_b:
            doms_inv[n].append(b)

    return dom_tree_aux(block_name(blocks[0]), doms_inv)

def print_edges(tree):
    if tree.children:
        for c in tree.children:
            print(tree.root + " -> " + c.root + ";")

def print_tree(name, tree):
     print("digraph " + name + " {")
     print_edges(tree)
     print("}")

def dom_frontier(func):
    return None
