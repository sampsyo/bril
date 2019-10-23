#!/usr/bin/env node
import * as df from './dataflow';
import * as cf from './controlflow';
import { readStdin } from './util';
import * as b from './bril';
import { Ident, Function, Instruction, EffectOperation } from './bril';

function getTrace(
  startLabel: Ident,
  funcMap: cf.FuncMap,
  cfg: Map<Ident, cf.CFGStruct>,
  onBranch: (op: EffectOperation) => boolean,
  blocks: number
): Ident[] {
  let remaining = blocks;
  // let trace: Instruction[] = [];
  let trace: Ident[] = [];
  let curLabel = startLabel;

  let liveOuts = getLiveOut(funcMap, cfg);

  while (--remaining > 0) {
    let instrs = funcMap.blocks.get(curLabel);
    let node = cfg.get(curLabel);
    if (!instrs || !node) throw `${curLabel} not a basic block`;

    // prevent multiple entries to the trace
    if (node.preds.length > 1 && trace.length !== 0) return trace;

    // not a conditional jump, add the this block to the trace
    if (node.succ.length === 1) {
      trace.push(curLabel);
      curLabel = node.succ[0].label;
    } else { // conditional jump
      let condInstr = instrs[instrs.length - 1];
      if (!('op' in condInstr && condInstr.op === 'br')) throw `Expected a branch but found ${condInstr}`;

      let failLabel = "";
      let nextLabel = "";
      if (onBranch(condInstr)) {
        nextLabel = condInstr.args[1];
        failLabel = condInstr.args[2];
      }
      else {
        nextLabel = condInstr.args[2];
        failLabel = condInstr.args[1];
      }
      let live = liveOuts.get(nextLabel);
      if (!live) throw new Error(`${nextLabel} not in liveOuts`);
      // replace condition with trace instruction
      let traceInstr: b.TraceEffectOperation = {
        op: "trace",
        failLabel,
        effect: condInstr,
        args: [...live.values(), condInstr.args[0]],
      };
      instrs.pop();
      instrs.push(traceInstr);
      // condInstr = traceInstr;
      trace.push(curLabel);
      curLabel = nextLabel;
    }
  }

  return trace;
}

function transfer(block: b.Instruction[], liveOut: Set<b.Ident>): Set<b.Ident> {
  let liveIn: Set<b.Ident> = new Set();
  liveOut.forEach(v => liveIn.add(v));

  for (let i = block.length - 1; i >= 0; i--) {
    let instr = block[i];
    if ('dest' in instr) {
      liveIn.delete(instr.dest);
    }
    if ('op' in instr) {
      switch (instr.op) {
        case "br":
          liveIn.add(instr.args[0]);
          break;
        case "const":
          break;
        case "print":
          instr.args.forEach(a1 => liveIn.add(a1));
          break;
        default:
          // value operation
          if ('dest' in instr) instr.args.forEach(a1 => liveIn.add(a1));

      }
    }
  }
  return liveIn;
}

function getLiveOut(func: cf.FuncMap, cfg: Map<b.Ident, cf.CFGStruct>): Map<b.Ident, Set<b.Ident>> {
  let ins: Map<b.Ident, Set<b.Ident>> = new Map();
  let outs: Map<b.Ident, Set<b.Ident>> = new Map();
  let worklist: b.Ident[] = Array.from(func.blocks.keys());

  function getOrCreate(map: Map<b.Ident, Set<b.Ident>>, key: b.Ident): Set<b.Ident> {
    let val = map.get(key);
    if (!val) {
      val = new Set();
      map.set(key, val);
    }
    return val;
  }

  function union<T>(s1: Set<T>, s2: Set<T>) {
    s2.forEach(v => s1.add(v));
  }

  function equals<T>(s1: Set<T>, s2: Set<T>): boolean {
    let res = true;
    s1.forEach(v => res = res && s2.has(v));
    s2.forEach(v => res = res && s1.has(v));
    return res;
  }

  while (worklist.length !== 0) {
    let block = worklist.pop();
    if (!block) throw new Error("Worklist was empty?");
    let node = cfg.get(block);
    if (!node) throw new Error("Node undefined");

    let liveOuts: Set<b.Ident> = getOrCreate(outs, block);
    node.succ.forEach(v => union(liveOuts, getOrCreate(ins, v.label)));
    outs.set(block, liveOuts);

    let instrs = func.blocks.get(block);
    if (!instrs) throw new Error("Empty block");

    let liveIns: Set<b.Ident> = transfer(instrs, liveOuts);

    // if changed, update worklist
    let myIns = getOrCreate(ins, block);
    if (!equals(myIns, liveIns)) {
      ins.set(block, liveIns);
      node.preds.forEach(v => {
        if (!worklist.includes(v.label)) worklist.push(v.label);
      });
    }
  }

  return outs;
}

function traceInstrs(trace: b.Ident[], fm: cf.FuncMap): b.Instruction[] {
  let instrs: b.Instruction[] = [];
  for (let lbl of trace) {
    let nextInstrs = fm.blocks.get(lbl);
    if (!nextInstrs) throw `${lbl} not a basic block`;
    instrs = [...instrs, ...nextInstrs];
  }
  return instrs;
}

let SUPERBLOCK_IDX: number = 0;
function freshLabel(): string {
  let label = `superblock.${SUPERBLOCK_IDX}`;
  SUPERBLOCK_IDX++;
  return label;
}

function insertSuperblock(trace: b.Ident[], fm: cf.FuncMap, cfg: cf.CFG, newInstrs: b.Instruction[]): cf.FuncMap {
  let label: b.Ident = freshLabel();

  // XXX(what if the last instruction is a conditional?)
  // figure out where the superblock should jump
  let lastNode = cfg.get(trace[trace.length - 1]);
  if (!lastNode) throw `${trace[trace.length - 1]} was not in the CFG`;
  if (lastNode.succ.length === 0) {
    let ret: b.EffectOperation = {
      op: "ret",
      args: [],
    }
    newInstrs.push(ret);
  }
  if (lastNode.succ.length === 1) {
    let lbl: string = lastNode.succ[0].label;
    if (trace.includes(lbl)) {
      lbl = label;
    }
    let jmp: b.EffectOperation = {
      op: "jmp",
      args: [lbl],
    }
    newInstrs.push(jmp);
  } else {
    throw "Haven't dealt with conditional branches at ends of superblocks yet";
  }

  // add new block for superblock
  fm.blocks.set(label, newInstrs);

  // patch all jumps pointing to head of trace
  let headNode = cfg.get(trace[0]);
  if (!headNode) throw `${trace[0]} was not in the CFG`;
  let headLabel = headNode.label;
  for (let pred of headNode.preds) {
    let block = fm.blocks.get(pred.label);
    if (!block) throw `${pred.label} not a basic block`;
    let cond = block.pop();
    if (!cond) throw `block was empty`;
    if ('args' in cond) {
      cond.args = cond.args.slice().map(l => {
        if (l === headLabel) return label;
        else return l;
      })
    }
    block.push(cond);
  }

  // remove trace blocks
  trace.forEach(lbl => fm.blocks.delete(lbl));

  return fm;
}

function funcMapInstrs(fm: cf.FuncMap): (b.Instruction | b.Label)[] {
  let instrs: (b.Instruction | b.Label)[] = [];
  for (let [lbl, ins] of fm.blocks) {
    let label: b.Label = {
      label: lbl
    };
    instrs = [...instrs, label, ...ins]
  }
  return instrs;
}

function run(prog: b.Program): b.Program {
  let functions: Function[] = [];
  for (let func of prog.functions) {

    // let pfunc = patchFunction(func);
    let fm = cf.genFuncMap(func);
    let control = cf.getCFG(fm);
    let liveVars = getLiveOut(fm, control);
    // console.log(liveVars);

    // console.log("func map", control);
    let trace = getTrace("for.cond.2", fm, control, (_) => true, 10);
    let instrs = traceInstrs(trace, fm);
    // console.log("TRACE")
    // b.logInstrs(instrs);
    // console.log("END")

    let dag = df.dataflow(instrs);
    df.assignDagPriority(dag);
    // // console.dir(dag, { depth: 4 });
    let sched = df.listSchedule(dag, (is, _c) => is.length + 1 < 4);

    let newFm = insertSuperblock(trace, fm, control, sched);
    let newInstrs: (b.Instruction | b.Label)[] = funcMapInstrs(newFm);
    // let newInstrs: b.Instruction[] = [];
    // newFm.blocks.forEach((ins, _) => newInstrs = [...newInstrs, ...ins]);

    let newFunc: Function = {
      name: func.name,
      instrs: newInstrs,
    }

    functions.push(newFunc);
  }

  return { functions };
}

async function main() {
  let prog = JSON.parse(await readStdin()) as b.Program;
  console.log(JSON.stringify(run(prog)));
}

// Make unhandled promise rejections terminate.
process.on('unhandledRejection', e => { throw e });

main();
