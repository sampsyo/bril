export type WeightedCallGraph = Map<string, number>; // Map from A -> B edges to weights
const edge_sep = " -> ";

export function getEdge(graph: WeightedCallGraph, from: string, to: string): string {
  return from + edge_sep + to;
}

export function getVerticesFromEdge(edge: string): string[] {
  return edge.split(edge_sep);
}

export function incrementEdge(graph: WeightedCallGraph, from: string, to: string) {
  let edge = getEdge(graph, from, to);
  let curr = graph.get(edge);
  curr = curr === undefined ? 0 : curr;
  graph.set(edge, curr + 1);
}
