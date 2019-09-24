"""Form a basic-block-based control-flow graph for a Bril function and
emit a pydot generated image.
"""

# To read in files from json format
import json
import sys
#sys.path.append("examples")

# To generate diagrams
import pydot

# To create blocks 
from form_blocks import form_blocks

# cfg support functions
from cfg import block_map, successors, add_terminators

def generate_edges(graph):
    edges = []

    # for each node in graph
    for node in graph:

        # for each neighbour node of a single node
        for neighbour in graph[node]:
            # if edge exists then append
            edges.append((node, neighbour))
    return edges

def print_cfg(cfg):
    edges = generate_edges(cfg)

    for edge in edges:
        print('edge from {} -> {}'.format(edge[0], edge[1]))

def cfg_pydot(graph):
    tree = pydot.Dot(graph_type='graph')

    for node in graph:
        for neighbour in graph[node]:
            edge = pydot.Edge("block {}".format(node), "block {}".format(neighbour))
            tree.add_edge(edge)

    tree.write_png('example1_graph.png')

def draw_cfg(cfg):
    edges = generate_edges(cfg)

    cfg_pydot(cfg)

def form_cfg(bril, verbose):
    """Generate a GraphViz "dot" file showing the control flow graph for
    a Bril program.

    In `verbose` mode, include the instructions in the vertices.
    """
    cfg = {}

    for func in bril['functions']:
        blocks = block_map(form_blocks(func['instrs']))

        # Insert terminators into blocks that don't have them.
        add_terminators(blocks)

        # Add the control-flow edges.
        for i, (name, block) in enumerate(blocks.items()):
            if verbose:
                import vriltxt
                block_name = r'  {} [shape=box, xlabel="{}", label="{}\l"];'.format(
                    name,
                    name,
                    r'\l'.join(vriltxt.instr_to_string(i) for i in block),
                )
            else:
                block_name = '{}'.format(name)
            edges = []
            succ = successors(block[-1])
            for label in succ:
                edges.append('{}'.format(label))
            cfg.update({ block_name : edges })
            
    print_cfg(cfg)
    draw_cfg(cfg)

if __name__ == '__main__':
    form_cfg(json.load(sys.stdin), '-v' in sys.argv[1:])
