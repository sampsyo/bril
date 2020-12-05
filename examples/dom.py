import json
import sys

from cfg import block_map, successors, add_terminators, add_entry
from form_blocks import form_blocks


def map_inv(succ):
    """Invert a multimap.

    Given a successor edge map, for example, produce an inverted
    predecessor edge map.
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
    pred = map_inv(succ)
    nodes = list(reversed(postorder(succ, entry)))  # Reverse postorder.

    dom = {v: set(nodes) for v in succ}

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


def dom_fronts(dom, succ):
    """Compute the dominance frontier, given the dominance relation.
    """
    dom_inv = map_inv(dom)

    frontiers = {}
    for block in dom:
        # Find all successors of dominated blocks.
        dominated_succs = set()
        for dominated in dom_inv[block]:
            dominated_succs.update(succ[dominated])

        # You're in the frontier if you're not strictly dominated by the
        # current block.
        frontiers[block] = [b for b in dominated_succs
                            if b not in dom_inv[block] or b == block]

    return frontiers


def dom_tree(dom):
    # Get the blocks strictly dominated by a block strictly dominated by
    # a given block.
    dom_inv = map_inv(dom)
    dom_inv_strict = {a: {b for b in bs if b != a}
                      for a, bs in dom_inv.items()}
    dom_inv_strict_2x = {a: set().union(*(dom_inv_strict[b] for b in bs))
                         for a, bs in dom_inv_strict.items()}
    return {
        a: {b for b in bs if b not in dom_inv_strict_2x[a]}
        for a, bs in dom_inv_strict.items()
    }


def print_dom(bril, mode):
    for func in bril['functions']:
        blocks = block_map(form_blocks(func['instrs']))
        add_entry(blocks)
        add_terminators(blocks)
        succ = {name: successors(block[-1]) for name, block in blocks.items()}
        dom = get_dom(succ, list(blocks.keys())[0])

        if mode == 'front':
            res = dom_fronts(dom, succ)
        elif mode == 'tree':
            res = dom_tree(dom)
        else:
            res = dom

        # Format as JSON for stable output.
        print(json.dumps(
            {k: sorted(list(v)) for k, v in res.items()},
            indent=2, sort_keys=True,
        ))


if __name__ == '__main__':
    print_dom(
        json.load(sys.stdin),
        'dom' if len(sys.argv) < 2 else sys.argv[1]
    )
