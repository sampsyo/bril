#!/usr/bin/env node
import * as df from './dataflow';
import * as cf from './controlflow';
import { readStdin } from './util';
import * as bril from './bril';
import { Ident, Function, Instruction, EffectOperation } from './bril';

/**
 * Given a sequence of instructions, generate a group with pre-condition tests
 * extracted out from the function.
 */
function toGroup(trace: bril.MicroInstruction[], failLabel: bril.Ident): bril.Group {
  let conds: bril.ValueOperation[] = [];
  let instrs: (bril.ValueOperation | bril.Constant)[] = [];

  // Calculate set of pre conditions for trace
  for (let inst of trace.slice().reverse()) {
    let condArgs: Set<bril.Ident> = new Set();
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
  labelMap: Map<Ident, Instruction[]>,
  predMap: Map<Ident, Ident[]>,
  onBranch: (op: EffectOperation) => boolean,
  blocks: number
): Instruction[] {
  let remaining = blocks;
  let trace: Instruction[] = [];
  let curLabel = startLabel;

  while (--remaining > 0) {
    // If there more than one entrance to this block, don't add it to the
    // trace.
    let preds = predMap.get(curLabel);
    if (preds && preds.length > 1 && trace.length !== 0) {
      return trace;
    }
    let instrs = labelMap.get(curLabel);

    if (!instrs) {
      throw new Error(`Unknown label ${curLabel}`)
    }
    console.log(instrs)

    trace.push(...instrs.slice(0, -1));
    let final = instrs[instrs.length - 1];

    if ('op' in final) {
      switch (final.op) {
        case "br": {
          if (onBranch(final)) curLabel = final.args[1];
          else curLabel = final.args[2];
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

function patchFunction(func: bril.Function): bril.Function {
  let idx = 0;
  // Find the first block's label.
  let curInst = func.instrs[idx];
  while (!('label' in curInst)) {
    idx++;
    curInst = func.instrs[idx];
  }
  let first = func.instrs.slice(0, idx);
  let last = func.instrs.slice(idx, func.instrs.length);
  let jmp: bril.EffectOperation = {
    op: "jmp", args: [curInst.label]
  }
  let start: bril.Label = {label: 'start'}
  return {
    name: func.name,
    instrs: [start, ...first, jmp, ...last ]
  }
}

function simple(prog: bril.Program): Array<bril.Instruction> {
  for (let func of prog.functions) {

    let pfunc = patchFunction(func);
    let fm = cf.genFuncMap(pfunc);
    let control = cf.getPreds(fm);

    let trace = getTrace("for.cond.2", fm, control, (_) => true, 10);
    console.log("TRACE")
    console.log(trace);
    console.log("END")

    if (func.name === "main") {
      let res: bril.Instruction[] = [];
      for (let ins of func.instrs) {
        if ("instrs" in ins || "op" in ins) {
          res.push(ins);
        }
      }
      return res;
    }
  }
  return [];
}

async function main() {
  let prog = JSON.parse(await readStdin()) as bril.Program;
  let trace = simple(prog);
  let dag = df.dataflow(trace);
  df.assignDagPriority(dag);
  let sched = df.listSchedule(dag, (is, _c) => is.length + 1 < 4);
  console.dir(sched, { depth: 3 });
}

// Make unhandled promise rejections terminate.
process.on('unhandledRejection', e => { throw e });

main();
