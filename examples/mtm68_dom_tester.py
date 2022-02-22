import sys
import json
from mtm68_dom import print_tree, dom_tree, find_doms, dom_frontier, all_paths

def assert_doms(paths, doms):
    for k, v in doms.items():
        for b in v:
            for path in paths:
                if k in path:
                    assert b in path

def assert_tree(tree, doms):
    if tree != None:
        # Assert tree immediatley dominates all children
        for c in tree.children:
            assert c.root in doms[c.root]
        # Assert holds for rest of tree
        for c in tree.children:
            assert_tree(c, doms)

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
