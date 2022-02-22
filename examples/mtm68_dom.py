import sys
import json

from form_blocks import form_blocks, block_name
from mtm68_cfg import Cfg

def all_paths_s_t(cfg, path, s, t):
    # If we are at t, we are done constructing path
    if block_name(s) == block_name(t):
        return [path]

    # Otherwise we must search for path using successors
    all_path_lst = []
    succs = cfg.get_succ(block_name(s))
    for succ in succs:
        # Create new path to add to so we can mutate easily
        new_path = path[:]
        succ_n = block_name(succ)

        # Do not follow cycles
        if succ_n not in new_path:
            new_path.append(succ_n)
            l = all_paths_s_t(cfg, new_path, succ, t)
            all_path_lst.extend(l)

    return all_path_lst

def all_paths(func):
    blocks = list(form_blocks(func['instrs']))
    cfg = Cfg(blocks)
    all_paths_lst = []
    no_succs = []
    for block in blocks:
        if len(cfg.get_succ(block_name(block))) == 0:
            no_succs.append(block)
    first_b = blocks[0]
    for exit in no_succs:
            l = all_paths_s_t(cfg, [block_name(first_b)], first_b, exit)
            all_paths_lst.extend(l)
    return all_paths_lst

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

def dom_tree_aux(name, doms):
    if not doms[name]:
        return Tree(name, [])
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

def dom_frontier_aux(dom_fr, cfg, tree):
    if not tree:
        return set()
    else:
        succs = set(map(block_name, cfg.get_succ(tree.root)))
        children = set() if not tree.children else tree.children
        front = succs.difference(children)
        dom_fr[tree.root] = front
        for c in tree.children:
            dom_frontier_aux(dom_fr, cfg, c)


def dom_frontier(func):
    blocks = list(form_blocks(func['instrs']))
    cfg = Cfg(blocks)
    tree = dom_tree(func)
    dom_fr = { block_name(block): set() for block in blocks }

    dom_frontier_aux(dom_fr, cfg, tree)
    return dom_fr


