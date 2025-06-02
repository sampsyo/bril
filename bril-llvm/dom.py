#!/usr/bin/python3

from brilpy import CFG
from functools import reduce


class Dominators:
    def __init__(self, func):
        g = CFG(func)

        # First compute dominators
        # IMPORTANT: This computes, for each block, the set of blocks that dominate
        # it, not the other way around
        self.doms = []
        self.doms.append(set([0]))  # Entry block is special, it's its own dominator
        for i in range(1, g.n):
            self.doms.append(set(range(g.n)))

        order = g.rpo()

        changed = True
        while changed:
            changed = False
            for i in order[1:]:  # no one can dominate 0 except 0
                d = {i}
                if g.preds[i]:
                    d |= reduce(
                        set.intersection,
                        [self.doms[p] for p in g.preds[i]],
                        set(range(g.n)),
                    )

                if d != self.doms[i]:
                    changed = True
                    self.doms[i] = d

        # Compute the "other way around" (from above), that is, for each block, the
        # set of blocks this block dominates
        self.dom_by = []
        for i in range(g.n):
            self.dom_by.append(set())

        for i, d in enumerate(self.doms):
            for mbr in d:
                self.dom_by[mbr].add(i)

        # Compute the dominance tree
        dt_parent = [None]

        for i in range(1, g.n):
            for j in self.doms[i]:
                if i != j:  # j strictly dominates i
                    immed_dom = True
                    for k in range(g.n):
                        if (
                            k != j
                            and k != i
                            and j in self.doms[k]
                            and k in self.doms[i]
                        ):
                            immed_dom = False
                            break
                    if immed_dom:
                        dt_parent.append(j)
                        break

        self.dom_tree = {}
        for i, p in enumerate(dt_parent):
            if p in self.dom_tree:
                self.dom_tree[p].append(i)
            else:
                self.dom_tree[p] = [i]

        # Compute dominance frontier
        self.frontier = []
        for i in range(g.n):
            self.frontier.append(set())

        for i, d in enumerate(self.doms):
            # Union of dominators for this node's preds
            pre_doms = reduce(set.union, [self.doms[p] for p in g.preds[i]], set())
            # Subtract out strict dominators for this node
            pre_doms = pre_doms.difference(self.doms[i].difference(set([i])))

            # This node is in the frontier for the remaining nodes:
            for p in pre_doms:
                self.frontier[p].add(i)
