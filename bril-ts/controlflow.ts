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

export type CFGStruct = {
  label: b.Ident;
  succ: CFGStruct[];
  preds: CFGStruct[];
}

export type CFG = Map<b.Ident, CFGStruct>;

/**
 * Mapping from a block label to its predecessors.
 */
export function getCFG(funcMap: FuncMap): CFG {
  let start = { label: 'start', succ: [], preds: [], }
  let cfg: CFGStruct = start
  let map: CFG = new Map();
  map.set('start', start)

  function getOrCreate(key: b.Ident): CFGStruct {
    let val = map.get(key);
    if (!val) {
      val = { label: key, succ: [], preds: [] };
      map.set(key, val);
    }
    return val
  }

  for (let [label, instrs] of funcMap.blocks) {
    let last = instrs[instrs.length - 1];
    let node: CFGStruct = getOrCreate(label);
    if (!last || !('op' in last)) {
      throw new Error("Block is empty");
    }
    switch (last.op) {
      case "br": {
        let node1 = getOrCreate(last.args[1]);
        let node2 = getOrCreate(last.args[2]);
        node.succ.push(node1, node2);
        node1.preds.push(node);
        node2.preds.push(node);
        break;
      }
      case "jmp": {
        let node1 = getOrCreate(last.args[0]);
        node.succ.push(node1);
        node1.preds.push(node);
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
          let node1 = getOrCreate(nextLabel);
          node.succ.push(node1);
          node1.preds.push(node);
          // add jump to next label
          let jmp: b.EffectOperation = {
            op: 'jmp',
            args: [nextLabel]
          }
          instrs.push(jmp);
        } else {
          let ret: b.EffectOperation = {
            op: 'ret',
            args: [],
          }
          instrs.push(ret);
        }
      }
    }
  }
  return map;
}
