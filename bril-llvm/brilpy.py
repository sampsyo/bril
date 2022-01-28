#!/usr/bin/python3

# Mark Moeller:

import sys

TERM = 'jmp', 'br', 'ret'


# From Lesson 2
def form_blocks(body):
    cur_block = []

    for inst in body:
        if 'op' in inst:
            cur_block.append(inst)

            # check for term
            if inst['op'] in TERM:
                yield cur_block
                cur_block = []

        else:  # label
            if len(cur_block) != 0:
                yield cur_block

            cur_block = [inst]

    if len(cur_block) != 0:
        yield cur_block


class CFG:
    # Constructs a new cfg (names, blocks, edges), where:
    # names: a list of block names
    # blocks: the list of blocks themselves
    # edges: idx->idx map of successors
    def __init__(self, func):
        self.names = []
        self.blocks = []

        # Map label -> block idx for that label
        labels = {}

        # Edges of the CFG
        self.edges = []

        # When we encounter jumps to labels that haven't appeared yet, add the
        # label here with a list of blocks that need to jump TO that label
        # label -> [list of blocks forward-jumping to it]
        resolve = {}

        def make_edge(idx, label):
            if label in labels:
                self.edges[idx].append(labels[label])
            else:
                if label in resolve:
                    resolve[label].append(idx)
                else:
                    resolve[label] = [idx]

        for i, block in enumerate(form_blocks(func['instrs'])):

            self.blocks.append(block)
            self.edges.append([])

            name = "b" + str(i)

            if 'label' in block[0]:
                name = block[0]['label']
                labels[name] = i

            self.names.append(name)

            if 'op' in block[-1] and (block[-1]['op'] == 'br' or
                                      block[-1]['op'] == 'jmp'):
                for label in block[-1]['labels']:
                    make_edge(i, label)

            elif 'op' in block[-1] and block[-1]['op'] != 'ret':
                self.edges[i] = [i+1]

        self.n = len(self.names)

        for lab, idcs in resolve.items():
            for idx in idcs:
                self.edges[idx].append(labels[lab])

        # If we added i+1 for the last block, remove it (there is no successor)
        if self.n in self.edges[-1]:
            self.edges[-1] = []

        # compute edges_r to get predecessors
        self.preds = []
        for i in range(self.n):
            self.preds.append([])

        for k, v in enumerate(self.edges):
            for d in v:
                self.preds[d].append(k)

    # perform a dfs in the specified order, calling pre(i) and post(i) upon
    # previsit and posvisit of i, respectively.
    # next_tree is called with no args after each time dfs_visit finishes a
    # connected component.
    def dfs(self, order=None, pre=None, post=None, next_tree=None, edges=None):

        if not order:
            order = list(range(self.n))

        if not edges:
            edges = self.edges

        WHITE = 0
        GRAY = 1
        BLACK = 2

        colors = [WHITE] * self.n

        def dfs_visit(node):
            if colors[node] == WHITE:
                colors[node] = GRAY
                if pre:
                    pre(node)
                for v in edges[node]:
                    dfs_visit(v)
                colors[node] = BLACK
                if post:
                    post(node)

        for i in order:
            dfs_visit(i)
            if next_tree:
                next_tree()

    # Return the indices in reverse-post-order.
    def rpo(self):
        visited = []

        def post_visit(i):
            visited.append(i)

        self.dfs(post=post_visit)
        visited.reverse()
        return visited

    # Unused first attempt. Computes SCCs in the graph.
    def natural_loops(self):

        sccs = []
        cur = []

        def postv(i):
            nonlocal cur
            cur.append(i)

        def nt():
            nonlocal cur
            if cur:
                sccs.append(cur)
                cur = []

        self.dfs(order=self.rpo(), post=postv, next_tree=nt, edges=self.preds)

        nl_list = []

        for cand in sccs:
            if len(cand) > 1:
                nat = True
                header = -1
                for b in cand:
                    for p in self.preds[b]:
                        if p not in cand:
                            if header == -1:
                                header = b
                            else:
                                nat = False
                                break
                if nat:
                    cand.remove(header)
                    nl_list.append([header] + cand)

        return nl_list

    def to_dot(self):
        s = "digraph g {\n"

        for u, nbrs in enumerate(self.edges):
            for v in nbrs:
                s += (self.names[u].replace('.', '_') + " -> " +
                      self.names[v].replace('.', '_') + ";\n")

        s += "}\n"
        return s

    def print_names(self):
        for i, n in enumerate(self.names):
            print("{} {}".format(i, n))


# ------------------------------------------------------------------------------
# Dataflow functions for SSA Reaching Definitions
#   Since we assume SSA, we can map from varname->single block defining
# ------------------------------------------------------------------------------

def rd_init(func, graph):
    in_b = []
    out_b = []

    in_b.append({})
    out_b.append({})

    if 'args' in func:
        for arg in func['args']:
            in_b[0][arg['name']] = 0

    for i in range(graph.n - 1):
        in_b.append({})
        out_b.append({})
    return (in_b, out_b)


def rd_xfer(in_b, block, idx):

    out_b = in_b.copy()

    for i, inst in enumerate(block):
        if 'dest' in inst:
            if inst['dest'] in out_b and out_b[inst['dest']] != idx:
                print(
                    "warning: illegal redef of var `{}`.".format(inst['dest'])
                    + "This function assumes SSA.",
                    file=sys.stderr
                )
            out_b[inst['dest']] = idx

    return out_b


def rd_merge(pred_list):

    result = {}

    for p in pred_list:
        for k, v in p.items():
            if k in result and v != result[k]:
                print("warning: illegal redef of var `{}` (multiple blocks).".format(v) +
                        " This function assumes SSA.", file=sys.stderr)
            result[k] = v

    return result



# ------------------------------------------------------------------------------
# Worklist function
# func: the function object (as loaded from json)
# init: (func, graph) -> (in_b, out_b): Computes initial datastructures. in_b
#                                       and out_b are each arrays of size
#                                       len(blocks).
# xfer: (in_b, block) -> out_b:         Compute transfer for a single block.
# merge: List of out_b -> in_b:         Given a list of predecessors' out_b's,
#                                       compute a single in_b.
# ------------------------------------------------------------------------------

def run_worklist(func, init, xfer, merge):
    graph = CFG(func)

    (in_b, out_b) = init(func, graph)

    worklist = list(range(graph.n))

    while worklist:
        b = worklist[0]
        worklist = worklist[1:]

        in_b[b] = merge([out_b[x] for x in graph.preds[b]]) if graph.preds[b] else {}

        out_b_copy = out_b[b].copy()

        out_b[b] = xfer(in_b[b], graph.blocks[b], b)

        if out_b[b] != out_b_copy:
            worklist += graph.edges[b]

    return (in_b, out_b)
