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

function run(prog: bril.Program) {
  for (let func of prog.functions) {

    // let pfunc = patchFunction(func);
    let fm = cf.genFuncMap(func);
    let control = cf.getPreds(fm);

    console.log("func map", control);
    let trace = getTrace("for.cond.2", fm.blocks, control, (_) => true, 10);
    console.log("TRACE")
    bril.logInstrs(trace);
    console.log("END")

    let dag = df.dataflow(trace);
    df.assignDagPriority(dag);
    let sched = df.listSchedule(dag, (is, _c) => is.length + 1 < 4);
    sched.forEach((v, i) => { console.log("GROUP", i); bril.logInstrs(v); console.log("END") })

    // if (func.name === "main") {
    //   let res: bril.Instruction[] = [];
    //   for (let ins of func.instrs) {
    //     if ("instrs" in ins || "op" in ins) {
    //       res.push(ins);
    //     }
    //   }
    //   return res;
    // }
  }
}

async function main() {
  let prog = JSON.parse(await readStdin()) as bril.Program;
  let trace = run(prog);
}

// Make unhandled promise rejections terminate.
process.on('unhandledRejection', e => { throw e });

main();
