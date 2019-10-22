import * as bril from './bril';

export type Node = {
  succs: Set<Node>,
  preds: Set<Node>,
  instr: bril.Instruction | "start"
};

export function dataflow(instrs: Array<bril.Instruction>): Node {
  let seen: Map<bril.Ident, Node> = new Map();
  let startNode: Node = { succs: new Set(), preds: new Set(), instr: "start" };
  // console.log(instrs)
  for (let instr of instrs) {
    let currNode: Node = { succs: new Set(), preds: new Set(), instr: instr };
    if ("args" in instr) {
      for (let arg of instr.args) {
        // we have already seen arg, add edge from seen[arg] -> currNode
        let parent = seen.get(arg);
        if (parent) {
          parent.succs.add(currNode);
          currNode.preds.add(parent);
        }
        // add edge from start -> currNode so that it's feasible to
        // schedule currNode.instr first
        startNode.succs.add(currNode);
      }
    }

    // add currNode to seen
    if ("dest" in instr) {
      seen.set(instr.dest, currNode);
    }
  }
  return startNode;
}
