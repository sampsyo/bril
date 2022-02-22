import sys
import json
from mtm68_dom import print_tree, dom_tree, find_doms, dom_frontier, all_paths

def assert_doms(paths, doms):
    for k, v in doms.items():
        for b in v:
            for path in paths:
                if k in path:
                    assert b in path

def dom_test(prog, arg):
    for func in prog['functions']:
        if arg == 'doms':
            print(func['name'])
            doms = find_doms(func)
            print(json.dumps(
                { k : sorted(list(v)) for k, v in doms.items()},
                indent=2, sort_keys=True
            ))
        elif arg == 'tree':
            tree = dom_tree(func)
            print_tree(func['name'], tree)
        elif arg == 'front':
            front = dom_frontier(func)
            print(json.dumps(
                { k : sorted(list(v)) for k, v in front.items()},
                indent=2, sort_keys=True
            ))
        elif arg == 'paths':
            paths = all_paths(func)
            doms = find_doms(func)
            assert_doms(paths, doms)
        else:
            print("Unknown arg")

if __name__ == '__main__':
    dom_test(json.load(sys.stdin), sys.argv[1])
