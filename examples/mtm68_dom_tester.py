import sys
import json
from form_blocks import form_blocks, block_name
from mtm68_dom import print_tree, dom_tree, find_doms, dom_frontier, all_paths
from mtm68_cfg import Cfg

def assert_doms(paths, doms):
    """
    We know that a block dominates another block
    when it is on all paths from the entry block
    to any finish blocks. We assert this property
    for all blocks.
    """
    for dominated_by, dominated in doms.items():
        for b in dominated:
            for path in paths:
                if dominated_by in path:
                    assert b in path

def assert_tree(tree, doms):
    """
    Assert that the tree is constructed correctly, i.e when
    forall node, forall childrenn, the root immediatley
    dominates the child.
    """
    if tree != None:
        # Assert tree immediatley dominates all children
        for c in tree.children:
            assert c.root in doms[c.root]
        # Assert holds for rest of tree
        for c in tree.children:
            assert_tree(c, doms)

def assert_front(front, doms, cfg):
    """
    Assert that the frontier is constructed correctly, i.e.
    when all succs to nodes in the dom tree, who are not
    part of the dom tree themselves are listed as on the
    dom frontier.
    """
    for k, v in front.items():
        for b in v:
            ret  = False
            for d in doms[k]:
                succs = cfg.get_succ(d)
                for succ in succs:
                    ret |= b == block_name(succ)
            assert ret

def test_doms(func):
    print(func['name'])
    doms = find_doms(func)
    paths = all_paths(func)
    assert_doms(paths, doms)
    print(json.dumps(
        { k : sorted(list(v)) for k, v in doms.items()},
        indent=2, sort_keys=True
    ))

def test_tree(func):
    tree = dom_tree(func)
    doms = find_doms(func)
    assert_tree(tree, doms)
    print_tree(func['name'], tree)

def test_front(func):
    front = dom_frontier(func)
    doms = find_doms(func)
    blocks = list(form_blocks(func['instrs']))
    cfg = Cfg(blocks)
    assert_front(front, doms, cfg)
    print(json.dumps(
        { k : sorted(list(v)) for k, v in front.items()},
        indent=2, sort_keys=True
    ))

def dom_test(prog, arg):
    for func in prog['functions']:
        if arg == 'doms':
            test_doms(func)
        elif arg == 'tree':
            test_tree(func)
        elif arg == 'front':
            test_front(func)
        else:
            print("Unknown arg")

if __name__ == '__main__':
    dom_test(json.load(sys.stdin), sys.argv[1])
