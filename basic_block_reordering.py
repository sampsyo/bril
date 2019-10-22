import collections
import copy
import json
import sys
sys.path.append('bril-txt/')
sys.path.append('examples/')
import briltxt
import cfg
import form_blocks



Edge = collections.namedtuple('Edge', ['s', 't'])

def coalesce(graph, coalesced_map):
    edges = sorted(graph, key=graph.get, reverse=True)  # decreasing order
    chains = [c for _, c in coalesced_map.items()]
    sources = [c[-1] for c in chains]
    targets = [c[0] for c in chains]
    for e in edges:
        if e.s in sources and e.t in targets:
            s_key, s_chain = [(k, v) for k, v in coalesced_map.items() if e.s in v][0]
            t_key, t_chain = [(k, v) for k, v in coalesced_map.items() if e.t in v][0]
            new_key = '{} -> {}'.format(s_key, t_key)
            new_chain = s_chain + t_chain
            coalesced_map.pop(s_key)
            coalesced_map.pop(t_key)
            coalesced_map[new_key] = new_chain
            return coalesced_map
    return None

def combine_chains(cmap, precedence):
    final_order = list()
    precedence = sorted(precedence, key=precedence.get, reverse=True)
    for s, t in precedence:
        if s in cmap and t in cmap:
            final_order.append(cmap.pop(s))
        if t in cmap:
            final_order.append(cmap.pop(t))
        if s in cmap and t not in cmap:
            final_order.insert(final_order.index(t), cmap.pop(s))
    if len(cmap) > 0:
        final_order.extend(cmap.values())
    return [block for chain in final_order for block in chain]

def json_to_graph(json_graph):
    edge_weights = dict()
    nodes = set()
    for edge in json_graph:
        e = Edge(edge['from'], edge['to'])
        nodes.add(edge['from'])
        nodes.add(edge['to'])
        if e.s != e.t:
            edge_weights[e] = edge_weights.get(e, 0) + edge['count']
    return edge_weights, nodes

if __name__ == '__main__':
    bril_file = sys.argv[1]
    profile_file = sys.argv[2]
    with open(bril_file) as f:
        program = json.loads(briltxt.parse_bril(f.read()))
    with open(profile_file) as f:
        basic_block_flows = json.load(f)['basic_block_flows']
    for function in program['functions']:
        function_name = function['name']
        block_edges = [e for e in basic_block_flows if e['function'] == function_name]
        if len(block_edges) == 0:
            continue
        graph, nodes = json_to_graph(block_edges)
        coalesced_map = {node: [node] for node in nodes}
        while True:
            new_map = coalesce(graph, coalesced_map)
            if not new_map:
                break
            coalesced_map = new_map
        precedence = dict()  # from (chain_name src, chain_name target), count)
        for name, chain in coalesced_map.items():
            for node in chain:
                dests = [(e.t, count) for e, count in graph.items() if e.s == node]
                for dest_name, dest_chain in coalesced_map.items():
                    if name == dest_name:
                        continue
                    for dest, w in dests:
                        if dest in dest_chain:
                            precedence[(name, dest_name)] = precedence.get((name, dest_name), 0) + 1
        basic_block_order = combine_chains(coalesced_map, precedence)
        # print(basic_block_order)
        blocks = cfg.block_map(form_blocks.form_blocks(function['instrs']))
        cfg.add_terminators(blocks)
        instrs = []
        preds, succs = cfg.edges(blocks)
        for i, label in enumerate(basic_block_order):
            if len(preds[label]) != 0:
                instrs.append({'label': label})
            if i < len(basic_block_order) - 1:
                for instr in blocks[label]:
                    if 'op' in instr and instr['op'] == 'jmp':
                        next_block = basic_block_order[i + 1]
                        if succs[label] == [next_block]:
                            preds[next_block].remove(label)
                            continue
                    instrs.append(instr)
                    
            else:
                instrs += blocks[label]
        function['instrs'] = instrs

    print(json.dumps(program, indent=2, sort_keys=True))
