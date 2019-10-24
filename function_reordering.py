import collections
import copy
import json
import sys
sys.path.append('bril-txt/')
import briltxt


Edge = collections.namedtuple('Edge', ['s', 't'])

def max_weight_edge(call_graph):
    return max(call_graph, key=call_graph.get)

def hottest_node(call_graph):
    nodes = dict()
    for e, count in call_graph.items():
        nodes[e.t] = nodes.get(e.t, 0) + count
    return max(nodes, key=nodes.get)

def json_to_graph(json_graph):
    edge_weights = dict()
    for edge in json_graph:
        e = Edge(edge['from'], edge['to'])
        if e.s != e.t:
            edge_weights[e] = edge_weights.get(e, 0) + edge['count']
    return edge_weights

def coalesce_nodes(hot_node, graph, coalesced_map):
    coalesced_map = copy.deepcopy(coalesced_map)
    preds = [(e.s, c) for e, c in graph.items() if e.t == hot_node]
    max_pred = max(preds, key= lambda x: x[1])
    a, b = max_pred[0], hot_node
    new_node_name = '{} -> {}'.format(a, b)
    assert new_node_name not in coalesced_map
    a_nodes = coalesced_map.get(a, [a])
    b_nodes = coalesced_map.get(b, [b])
    coalesced_map[new_node_name] = a_nodes + b_nodes
    coalesced_map.pop(a, None)
    coalesced_map.pop(b, None)
    new_edges = dict()  # tuple to count
    for edge, count in graph.items():
        from_node = new_node_name if edge.s == a or edge.s == b else edge.s
        to_node = new_node_name if edge.t == a or edge.t == b else edge.t
        e = Edge(from_node, to_node)
        if from_node != to_node:
            new_edges[e] = new_edges.get(e, 0) + count
    return new_edges, coalesced_map

def reorder(program, profile_out):
    call_graph = profile_out['call_graph']
    if len(call_graph) == 0:
        return program
    coalesced_map = dict()
    graph = json_to_graph(call_graph)
    while len(graph) > 0:
        hot_node = hottest_node(graph)
        graph, coalesced_map = coalesce_nodes(hot_node, graph, coalesced_map)
    assert len(coalesced_map) == 1
    function_order = coalesced_map[list(coalesced_map)[0]]
    new_program = copy.deepcopy(program)
    new_program['functions'] = sorted(program['functions'], key=lambda f: function_order.index(f['name']) if f['name'] in function_order else len(function_order))
    return new_program


if __name__ == '__main__':
    bril_file = sys.argv[1]
    profile_file = sys.argv[2]
    with open(bril_file) as f:
        program = json.loads(briltxt.parse_bril(f.read()))
    with open(profile_file) as f:
        profile_out = json.load(f)
    program = reorder(program, profile_out)
    briltxt.print_prog(program)
