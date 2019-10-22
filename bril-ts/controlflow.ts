import * as b from './bril';

/**
 * Generates a map from labels to the instructions run by jumping to label
 * i.e generated basic blocks.
 */
export function genFuncMap(func: b.Function): Map<b.Ident, b.Instruction[]> {
  let map: Map<b.Ident, b.Instruction[]> = new Map();

  let curLabel = "start";
  for (let instr of func.instrs) {
    // instr is a label
    if ('label' in instr) {
      curLabel = instr.label;
    } else {
      let block = map.get(curLabel);
      if (block) {
        block.push(instr);
      } else {
        map.set(curLabel, [instr]);
      }
    }
  }

  return map;
}


/**
 * Mapping from a block label to its predecessors.
 */
export function getPreds(funcMap: Map<b.Ident, b.Instruction[]>): Map<b.Ident, b.Ident[]> {
  let map: Map<b.Ident, b.Ident[]> = new Map();

  function addOrCreate(pred: b.Ident, me: b.Ident) {
    let val = map.get(pred);
    if (!val) {
      val = []
      map.set(pred, val)
    }
    val.push(me);
  }

  for (let [label, instrs] of funcMap) {
    let last = instrs[instrs.length - 1];
    if (!last || !('op' in last)) {
      throw new Error("Block is empty");
    }
    switch (last.op) {
      case "br": {
        addOrCreate(last.args[1], label);
        addOrCreate(last.args[2], label);
        break;
      }
      case "jmp": {
        addOrCreate(last.args[0], label);
        break;
      }
      case "ret": {
        continue;
      }
    }
  }
  return map;
}
