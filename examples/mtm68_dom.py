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
    doms = {block_name(block) : set(map(block_name, blocks)) for block in blocks}
    dom_changing = True
    while dom_changing:
        dom_changed = False
        for block in reverse_post_order:
            name = block_name(block)
            preds = cfg.get_pred(name)
            len_before = len(doms[name])
            doms[name] = {name}.union(intersection(doms, preds))
            len_after = len(doms[name])
            dom_changed |= len_before != len_after
        dom_changing = dom_changed
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
    if not doms[name]:
        return Tree(name, None)
    else:
        return Tree(name, [dom_tree_aux(c, doms) for c in doms[name]])


def dom_tree(func):
    blocks = list(form_blocks(func['instrs']))
    doms = find_doms(func)

    # Invert the mapping
    doms_inv = {block_name(block) : set() for block in blocks}
    for b, dominated_by_b in doms.items():
        for n in dominated_by_b:
            doms_inv[n].add(b)

    # Make it strict
    for k, v in doms_inv.items():
        v.remove(k)

    # Make it immediate
    for k1 in doms_inv.keys():
        for k2 in doms_inv.keys():
            if k1 == k2:
                continue
            else:
                doms_inv[k1] = doms_inv[k1].difference(doms_inv[k2])

    # Build the tree
    return dom_tree_aux(block_name(blocks[0]), doms_inv)

def print_edges(tree):
    if tree.children:
        for c in tree.children:
            print('"' + tree.root + '" -> "' + c.root + '";')
            print_edges(c)

def print_tree(name, tree):
     print("digraph " + name + " {")
     print_edges(tree)
     print("}")

def dom_frontier(func):
    return None
