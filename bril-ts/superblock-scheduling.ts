#!/usr/bin/env node
import * as df from './dataflow';
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


/**
 * Assumes that the starting block of the function also has a name.
 */
function genFuncMap(func: Function): Map<Ident, Instruction[]> {
  let map: Map<Ident, Instruction[]> = new Map();
  for (let i = 0; i < func.instrs.length; i++) {
    let curInst = func.instrs[i];
    if ('label' in curInst) {
      let instrs: Instruction[] = [];
      let label = curInst.label;

      curInst = func.instrs[i++];
      while (!('label' in curInst)) {
        instrs.push(curInst);
        curInst = func.instrs[i++];
      }
      map.set(label, instrs);
    }
  }

  return map;
}

function getTrace(
  startLabel: Ident,
  labelMap: Map<Ident, Instruction[]>,
  onBranch: (op: EffectOperation) => boolean,
  blocks: number): Instruction[] {
    let remaining = blocks;
    let trace: Instruction[] = [];
    let curLabel = startLabel;

    while(--remaining > 0) {
      let instrs = labelMap.get(curLabel);

      if (!instrs) {
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

function simple(prog: bril.Program): Array<bril.Instruction> {
  for (let func of prog.functions) {
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
