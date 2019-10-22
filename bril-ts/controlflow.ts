import * as b from './bril';

export type FuncMap = {
  blocks: Map<b.Ident, b.Instruction[]>,
  order: Map<b.Ident, number>
}

/**
 * Generates a map from labels to the instructions run by jumping to label
 * i.e generated basic blocks.
 */
export function genFuncMap(func: b.Function): FuncMap {
  let map: Map<b.Ident, b.Instruction[]> = new Map();
  let order: Map<b.Ident, number> = new Map();

  let curLabel = "start";
  let idx = 0;
  for (let instr of func.instrs) {
    // instr is a label
    if ('label' in instr) {
      curLabel = instr.label;
      idx++;
    } else {
      let block = map.get(curLabel);
      if (block) {
        block.push(instr);
      } else {
        order.set(curLabel, idx);
        map.set(curLabel, [instr]);
      }
    }
  }

  return {
    blocks: map,
    order: order
  };
}

/**
 * Mapping from a block label to its predecessors.
 */
export function getPreds(funcMap: FuncMap): Map<b.Ident, b.Ident[]> {
  let map: Map<b.Ident, b.Ident[]> = new Map();

  function addOrCreate(pred: b.Ident, me: b.Ident) {
    let val = map.get(pred);
    if (!val) {
      val = []
      map.set(pred, val)
    }
    val.push(me);
  }

  for (let [label, instrs] of funcMap.blocks) {
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
        break;
      }
      // fallthrough branch
      default: {
        let idx = funcMap.order.get(label);
        if (idx === undefined) throw new Error(`${label} didn't exist in block`);

        let nextLabel = undefined;
        for (let [lbl, i] of funcMap.order) {
          if (i === idx + 1) {
            nextLabel = lbl;
            break;
          }
        }
        if (nextLabel) {
          addOrCreate(nextLabel, label);
          // add jump to next label
          let jmp: b.EffectOperation = {
            op: 'jmp',
            args: [nextLabel]
          }
          instrs.push(jmp);
        }
      }
    }
  }
  return map;
}
