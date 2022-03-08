import sys
import json

from form_blocks import form_blocks, block_name
from mtm68_cfg import Cfg

def all_paths_s_t(cfg, path, s, t):
    """
    Returns all paths from s to t, built on top of
    path that does not create any cycles.
    """
    # If we are at t, we are done constructing path
    if block_name(s) == block_name(t):
        return [path]

    # Otherwise we must search for path using successors
    all_path_lst = []
    for succ in cfg.get_succ(block_name(s)):
        succ_n = block_name(succ)

        # Create new path to add to so we can mutate easily
        new_path = path[:]

        # Get all paths containing succ to t
        # As long as visiting t does not
        # not create any cycles
        if succ_n not in new_path:
            new_path.append(succ_n)
            l = all_paths_s_t(cfg, new_path, succ, t)
            all_path_lst.extend(l)

    return all_path_lst

def all_paths(func):
    """
    Returns all paths from the start block
    to all possible exit blocks, requiring that
    no path has a cycle
    """
    blocks = list(form_blocks(func['instrs']))
    cfg = Cfg(blocks)
    all_paths_lst = []

    exits = cfg.get_exits()
    first_b = blocks[0]

    # Build all paths
    for exit in exits:
        l = all_paths_s_t(cfg, [block_name(first_b)], first_b, exit)
        all_paths_lst.extend(l)
    return all_paths_lst

def intersection(doms, preds):
    """
    Returns the intersection of dom(p) forall pred p.
    """
    if not preds:
        return set()
    else:
        for j, pred in enumerate(preds):
            name = block_name(pred)
            # Need to add all in the first so we are not
            # initially intersecting with the empty set
            if j == 0:
                i = doms[name]
            else:
                i = i.intersection(doms[name])
        return i

def find_doms(func):
    blocks = list(form_blocks(func['instrs']))
    cfg = Cfg(blocks)
    reverse_post_order = cfg.reverse_post_order(blocks[0])
    doms = {block_name(block)
            : set(map(block_name, blocks)) for block in blocks}

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

def dom_imm(func):
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

    return doms_inv


def dom_tree(func):
    blocks = list(form_blocks(func['instrs']))
    doms_inv = dom_imm(func)

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
        children = set() if not tree.children else {t.root for t in tree.children}
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
