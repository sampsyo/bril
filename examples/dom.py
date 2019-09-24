import json
import sys

from cfg import block_map, successors, add_terminators
from form_blocks import form_blocks


def get_pred(succ):
    """Given a successor edge map, produce an inverted predecessor edge
    map.
    """
    out = {key: [] for key in succ}
    for p, ss in succ.items():
        for s in ss:
            out[s].append(p)
    return out


def postorder_helper(succ, root, explored, out):
    """Given a successor edge map, produce a list of all the nodes in
    the graph in postorder by appending to the `out` list.
    """
    if root in explored:
        return
    explored.add(root)

    for s in succ[root]:
        postorder_helper(succ, s, explored, out)
    out.append(root)


def postorder(succ, root):
    out = []
    postorder_helper(succ, root, set(), out)
    return out


def intersect(sets):
    sets = list(sets)
    if not sets:
        return set()
    out = set(sets[0])
    for s in sets[1:]:
        out &= s
    return out


def get_dom(succ, entry):
    pred = get_pred(succ)
    nodes = list(reversed(postorder(succ, entry)))  # Reverse postorder.

    dom = {v: set(nodes) for v in nodes}

    while True:
        changed = False

        for node in nodes:
            new_dom = intersect(dom[p] for p in pred[node])
            new_dom.add(node)

            if dom[node] != new_dom:
                dom[node] = new_dom
                changed = True

        if not changed:
            break

    return dom


def print_dom(bril):
    for func in bril['functions']:
        blocks = block_map(form_blocks(func['instrs']))
        add_terminators(blocks)
        succ = {name: successors(block[-1]) for name, block in blocks.items()}
        dom = get_dom(succ, list(blocks.keys())[0])
        print(dom)


if __name__ == '__main__':
    print_dom(json.load(sys.stdin))
