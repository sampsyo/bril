import sys
import json
from mtm68_dom import print_tree, dom_tree, find_doms, dom_frontier

def serialize_sets(obj):
    if isinstance(obj, set):
        return list(obj)
    return obj

def dom_test(prog, arg):
    for func in prog['functions']:
        if arg == 'doms':
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
        else:
            print("Unknown arg")

if __name__ == '__main__':
    dom_test(json.load(sys.stdin), sys.argv[1])
