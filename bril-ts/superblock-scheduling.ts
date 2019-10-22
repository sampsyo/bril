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
          // Add fake id instruction
          let fake: b.ValueOperation = {
            op: "id",
            args: getLives(labelMap.get(curLabel) || []),
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

function getUses(block: b.Instruction[]): Set<b.Ident> {
  let uses: Set<b.Ident> = new Set();
  for (let inst of block) {
    if ('args' in inst)
      inst.args.forEach(el => uses.add(el));
  }
  return uses;
}

function getDefs(block: b.Instruction[]): Set<b.Ident> {
  let def: Set<b.Ident> = new Set();
  for (let inst of block) {
    if ('dest' in inst)
      def.add(inst.dest);
  }
  return def;
}

function getLiveOut(func: FuncMap, cfg: cf.CFGStruct): Map<b.Ident, Set<b.Ident>> {
  let out: Map<b.Ident, Set<b.Ident>> = new Map();
  let worklist: b.Ident[] = [];

  function getOrCreate(key: b.Ident): Set<b.Ident> {
    let val = out.get(key);
    if (!val) {
      val = new Set();
      map.add(val)
    }
    return val;
  }

  while(worklist) {
    block = worklist.pop();
    preds = cfg.preds
  }
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
    console.log(control)

    //console.log("func map", control);
    //let trace = getTrace("for.cond.2", fm.blocks, control, (_) => true, 10);
    //console.log("TRACE")
    //b.logInstrs(trace);
    //console.log("END")

    //let dag = df.dataflow(trace);
    //df.assignDagPriority(dag);
    ////console.dir(dag, { depth: 4 })
    //let sched = df.listSchedule(dag, (is, _c) => is.length + 1 < 4);
    //sched.forEach((v, i) => { console.log("GROUP", i); b.logInstrs(v); console.log("END") })
  }
}

async function main() {
  let prog = JSON.parse(await readStdin()) as b.Program;
  let trace = run(prog);
}

// Make unhandled promise rejections terminate.
process.on('unhandledRejection', e => { throw e });

main();
