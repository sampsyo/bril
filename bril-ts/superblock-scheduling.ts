#!/usr/bin/env node
import * as df from './dataflow';
import * as cf from './controlflow';
import { readStdin } from './util';
import * as b from './bril';
import { Ident, Function, Instruction, EffectOperation } from './bril';

/**
 * Given a sequence of instructions, generate a group with pre-condition tests
 * extracted out from the function.
 */
function toGroup(trace: b.MicroInstruction[], failLabel: b.Ident): b.Group {
  let conds: b.ValueOperation[] = [];
  let instrs: (b.ValueOperation | b.Constant)[] = [];

  // Calculate set of pre conditions for trace
  for (let inst of trace.slice().reverse()) {
    let condArgs: Set<b.Ident> = new Set();
    // If an Effect Operation, add the arguments to set of condArgs
    if (!('dest' in inst)) {
      inst.args.forEach(a => condArgs.add(a));
    } else if (inst.op != 'const' && condArgs.has(inst.dest)) {
      conds.push(inst);
    } else {
      instrs.push(inst);
    }
  }

  return { conds, instrs, failLabel }
}

function getTrace(
  startLabel: Ident,
  funcMap: cf.FuncMap,
  cfg: Map<Ident, cf.CFGStruct>,
  onBranch: (op: EffectOperation) => boolean,
  blocks: number
): Instruction[] {

  let remaining = blocks;
  let trace: Instruction[] = [];
  let curLabel = startLabel;

  let liveOuts = getLiveOut(funcMap, cfg);

  while (--remaining > 0) {
    // If there more than one entrance to this block, don't add it to the
    // trace.
    let node = cfg.get(curLabel);
    if (node && node.preds.length > 1 && trace.length !== 0) {
      return trace;
    }
    let instrs = funcMap.blocks.get(curLabel);

    if (instrs === undefined) {
      throw new Error(`Unknown label ${curLabel}`)
    }

    trace.push(...instrs.slice(0, -1));
    let final = instrs[instrs.length - 1];

    if ('op' in final) {
      switch (final.op) {
        case "br": {
          if (onBranch(final)) curLabel = final.args[1];
          else curLabel = final.args[2];
          let live = liveOuts.get(curLabel);
          if (!live) throw new Error(`${curLabel} not in liveOuts`);
          // Add fake id instruction
          let fake: b.ValueOperation = {
            op: "id",
            // args: getLiveOut(labelMap.get(curLabel) || []),
            args: Array.from(live.values()),
            dest: "DO_NO_WRONG",
            type: 'int'
          };
          trace.push(fake);
          break;
        }
        case "jmp": {
          curLabel = final.args[0];
          break;
        }
        case "ret": {
          return trace;
        }
      }
    }
  }

  return trace;
}

function getLives(block: b.Instruction[]): b.Ident[] {
  let lives: b.Ident[] = [];
  for (let inst of block) {
    if ('failLabel' in inst) {
      getLives([...inst.conds, ...inst.instrs]).forEach(el => lives.push(el));
    } else if ('dest' in inst) {
      lives.push(inst.dest)
    }
  }
  return lives;
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

function removeGarbage(insts: b.Instruction[]): b.Instruction[] {
  let newInsts: b.Instruction[] = [];
  for (let inst of insts) {
    if (('op' in inst) && inst.op === 'id' && inst.args.length > 1) continue;
    newInsts.push(inst);
  }
  return newInsts;
}

function run(prog: b.Program) {
  for (let func of prog.functions) {

    // let pfunc = patchFunction(func);
    let fm = cf.genFuncMap(func);
    let control = cf.getCFG(fm);
    let liveVars = getLiveOut(fm, control);
    // console.log(liveVars);

    // console.log("func map", control);
    let trace = getTrace("for.cond.2", fm, control, (_) => true, 10);
    console.log("TRACE")
    b.logInstrs(trace);
    console.log("END")

    let dag = df.dataflow(trace);
    df.assignDagPriority(dag);

    dag.succs.forEach(m => {
      if (m.instr !== "start" && "dest" in m.instr) {
        if (m.instr.dest === "DO_NO_WRONG") {
          console.dir(m, { depth: 3 });
        }
      }
    });
    // console.dir(dag, { depth: 4 });

    let sched = df.listSchedule(dag, (is, _c) => is.length + 1 < 4);
    sched.forEach((v, i) => { console.log("GROUP", i); b.logInstrs(v); console.log("END") })
  }
}

async function main() {
  let prog = JSON.parse(await readStdin()) as b.Program;
  let trace = run(prog);
}

// Make unhandled promise rejections terminate.
process.on('unhandledRejection', e => { throw e });

main();
