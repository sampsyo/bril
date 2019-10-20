import collections
import copy
import json
import sys

Edge = collections.namedtuple('Edge', ['s', 't'])

def max_weight_edge(call_graph):
    return max(call_graph, key=call_graph.get)

def json_to_graph(json_graph):
    edge_weights = dict()
    for edge in json_graph:
        e = Edge(edge['from'], edge['to'])
        if e.s != e.t:
            edge_weights[e] = edge_weights.get(e, 0) + edge['count']
    return edge_weights

def coalesce_nodes(a, b, graph, coalesced_map):
    coalesced_map = copy.deepcopy(coalesced_map)
    a_to_b = [e for e in graph if e.s == a and e.t == b]
    b_to_a = [e for e in graph if e.s == b and e.t == a]

    a_to_b_weight = sum(graph.get(e) for e in a_to_b)
    b_to_a_weight = sum(graph.get(e) for e in b_to_a)
    if b_to_a_weight > a_to_b_weight:
        a, b = b, a  # We want a -> b to be the higher-weighted edge

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

if __name__ == '__main__':
    bril_file = sys.argv[1]
    profile_file = sys.argv[2]
    with open(bril_file) as f:
        program = json.load(f)
    with open(profile_file) as f:
        call_graph = json.load(f)
    coalesced_map = dict()
    graph = json_to_graph(call_graph)
    while len(graph) > 0:
        print(graph)
        print('\n')
        max_e = max_weight_edge(graph)
        graph, coalesced_map = coalesce_nodes(max_e.s, max_e.t, graph, coalesced_map)
    assert len(coalesced_map) == 1
    print('Function order:', coalesced_map[list(coalesced_map)[0]])
