#!/usr/bin/env node
import * as bril from './bril';
import { readStdin, unreachable } from './util';

const argCounts: { [key in bril.OpCode]: number | null } = {
  add: 2,
  mul: 2,
  sub: 2,
  div: 2,
  id: 1,
  lt: 2,

  le: 2,
  gt: 2,
  ge: 2,
  eq: 2,
  not: 1,
  and: 2,
  or: 2,
  print: null,  // Any number of arguments.
  br: 3,
  jmp: 1,
  ret: 0,
  nop: 0,
};

const constCost = 1;
const opCosts: { [key in bril.OpCode]: number } = {
  add: 1,
  mul: 1,
  sub: 1,
  div: 1,
  id: 1,
  lt: 1,
  le: 1,
  gt: 1,
  ge: 1,
  eq: 1,
  not: 1,
  and: 1,
  or: 1,
  print: 0,  // Any number of arguments.
  br: 1,
  jmp: 1,
  ret: 1,
  nop: 1,
};
const groupCost = 1;

type Value = boolean | BigInt;
type Env = Map<bril.Ident, Value>;

function get(env: Env, ident: bril.Ident) {
  let val = env.get(ident);
  if (typeof val === 'undefined') {
    throw `undefined variable ${ident}`;
  }
  return val;
}

/**
 * Ensure that the instruction has exactly `count` arguments,
 * throwing an exception otherwise.
 */
function checkArgs(instr: bril.Operation, count: number) {
  if (instr.args.length != count) {
    throw `${instr.op} takes ${count} argument(s); got ${instr.args.length}`;
  }
}

function getInt(instr: bril.Operation, env: Env, index: number) {
  let val = get(env, instr.args[index]);
  if (typeof val !== 'bigint') {
    throw `${instr.op} argument ${index} must be a number`;
  }
  return val;
}

function getBool(instr: bril.Operation, env: Env, index: number) {
  let val = get(env, instr.args[index]);
  if (typeof val !== 'boolean') {
    throw `${instr.op} argument ${index} must be a boolean`;
  }
  return val;
}

/**
 * The thing to do after interpreting an instruction: either transfer
 * control to a label, go to the next instruction, end the function, or
 * signal the failure of a Group cond instruction.
 */
type Action = { "label": bril.Ident } | { "next": true } | { "end": true };
let NEXT: Action = { "next": true };
let END: Action = { "end": true };

/**
 * Interpret a micro-instruction in a given environment, possibly updating the
 * environment. If the instruction branches to a new label, return that label;
 * otherwise, return "next" to indicate that we should proceed to the next
 * instruction or "end" to terminate the function.
 */
function evalMicroInstr(instr: bril.MicroInstruction, env: Env): [Action, number] {
  // Check that we have the right number of arguments.
  if (instr.op === 'trace') throw `Cannot interpret traces`;
  if (instr.op !== "const") {
    let count = argCounts[instr.op];
    if (count === undefined) {
      throw "unknown opcode " + instr.op;
    } else if (count !== null) {
      checkArgs(instr, count);
    }
  }

  switch (instr.op) {
    case "const":
      // Ensure that JSON ints get represented appropriately.
      let value: Value;
      if (typeof instr.value === "number") {
        value = BigInt(instr.value);
      } else {
        value = instr.value;
      }

      env.set(instr.dest, value);
      return [NEXT, constCost];

    case "id": {
      let val = get(env, instr.args[0]);
      env.set(instr.dest, val);
      return [NEXT, opCosts[instr.op]];
    }

    case "add": {
      let val = getInt(instr, env, 0) + getInt(instr, env, 1);
      env.set(instr.dest, val);
      return [NEXT, opCosts[instr.op]];
    }

    case "mul": {
      let val = getInt(instr, env, 0) * getInt(instr, env, 1);
      env.set(instr.dest, val);
      return [NEXT, opCosts[instr.op]];
    }

    case "sub": {
      let val = getInt(instr, env, 0) - getInt(instr, env, 1);
      env.set(instr.dest, val);
      return [NEXT, opCosts[instr.op]];
    }

    case "div": {
      let val = getInt(instr, env, 0) / getInt(instr, env, 1);
      env.set(instr.dest, val);
      return [NEXT, opCosts[instr.op]];
    }

    case "le": {
      let val = getInt(instr, env, 0) <= getInt(instr, env, 1);
      env.set(instr.dest, val);
      return [NEXT, opCosts[instr.op]];
    }

    case "lt": {
      let val = getInt(instr, env, 0) < getInt(instr, env, 1);
      env.set(instr.dest, val);
      return [NEXT, opCosts[instr.op]];
    }

    case "gt": {
      let val = getInt(instr, env, 0) > getInt(instr, env, 1);
      env.set(instr.dest, val);
      return [NEXT, opCosts[instr.op]];
    }

    case "ge": {
      let val = getInt(instr, env, 0) >= getInt(instr, env, 1);
      env.set(instr.dest, val);
      return [NEXT, opCosts[instr.op]];
    }

    case "eq": {
      let val = getInt(instr, env, 0) === getInt(instr, env, 1);
      env.set(instr.dest, val);
      return [NEXT, opCosts[instr.op]];
    }

    case "not": {
      let val = !getBool(instr, env, 0);
      env.set(instr.dest, val);
      return [NEXT, opCosts[instr.op]];
    }

    case "and": {
      let val = getBool(instr, env, 0) && getBool(instr, env, 1);
      env.set(instr.dest, val);
      return [NEXT, opCosts[instr.op]];
    }

    case "or": {
      let val = getBool(instr, env, 0) || getBool(instr, env, 1);
      env.set(instr.dest, val);
      return [NEXT, opCosts[instr.op]];
    }

    case "print": {
      let values = instr.args.map(i => get(env, i).toString());
      console.log(...values);
      return [NEXT, opCosts[instr.op]];
    }

    case "jmp": {
      return [{ "label": instr.args[0] }, opCosts[instr.op]];
    }

    case "br": {
      let cond = getBool(instr, env, 0);
      if (cond) {
        return [{ "label": instr.args[1] }, opCosts[instr.op]];
      } else {
        return [{ "label": instr.args[2] }, opCosts[instr.op]];
      }
    }

    case "ret": {
      return [END, opCosts[instr.op]];
    }

    case "nop": {
      return [NEXT, opCosts[instr.op]];
    }
  }
  unreachable(instr);
  throw `unhandled opcode ${(instr as any).op}`;
}

/**
 * Returns true if all the writes in group are to distinct locations
 * and false otherwise.
 */
function noConflicts(group: bril.Group): Boolean {
  // Set of registers written to.
  let reads: Set<bril.Ident> = new Set(group.conds);
  // Set of registers read from.
  let writes: Set<bril.Ident> = new Set();

  for (let instr of [...group.instrs]) {
    // Add all reads to set of reads
    if ('args' in instr) {
      instr.args.forEach(arg => reads.add(arg));
    }

    if (writes.has(instr.dest) || reads.has(instr.dest)) {
      return false;
    }

    writes.add(instr.dest);
  }
  return true;
}

function evalInstr(instr: bril.Instruction, env: Env): [Action, number] {
  if ('effect' in instr) {
    throw `Traces can not be interpreted`;
  } else if ('op' in instr) { // is a micro instruction
    let [act, cost] = evalMicroInstr(instr, env);
    return [act, cost];
  } else { // is a group
    if (noConflicts(instr)) {
      // Evaluate the pre condition for group instruction.
      let condVal = true;
      instr.conds.forEach(c => {
        let lookup = get(env, c);
        if (typeof lookup === 'boolean') {
          condVal = condVal && lookup;
        }
      });
      if (!condVal) return [{ "label": instr.failLabel }, groupCost]
      // for (let cond of instr.conds) {
      //   // Copy the instruction
      //   let condCopy = Object.assign({}, cond);
      //   cond.dest = "INVALID";
      //   evalMicroInstr(cond, env);
      //   // If the condition was false, the instruction failed.
      //   if (!get(env, "INVALID"))
      //     return [{ "label": instr.failLabel }, groupCost]
      // }
      for (let inst of instr.instrs) {
        evalMicroInstr(inst, env);
      }
      return [NEXT, groupCost];
    } else {
      throw `Group is not resource compatible`
    }
  }
}

function evalFunc(func: bril.Function): number {
  let env: Env = new Map();
  let totCost = 0;
  for (let i = 0; i < func.instrs.length; ++i) {
    let line = func.instrs[i];
    // if line is not a label
    if (!('label' in line)) {
      let [action, instrCost] = evalInstr(line, env);
      totCost += instrCost;

      if ('label' in action) {
        // Search for the label and transfer control.
        for (i = 0; i < func.instrs.length; ++i) {
          let sLine = func.instrs[i];
          if ('label' in sLine && sLine.label === action.label) {
            break;
          }
        }
        if (i === func.instrs.length) {
          throw `label ${action.label} not found`;
        }
      } else if ('end' in action) {
        return totCost;
      }
    }
  }
  return totCost;
}

function evalProg(prog: bril.Program) {
  let cost = 0;
  for (let func of prog.functions) {
    if (func.name === "main") {
      cost += evalFunc(func);
    }
  }
  console.error("Cost: ", cost);
}

async function main() {
  let prog = JSON.parse(await readStdin()) as bril.Program;
  evalProg(prog);
}

// Make unhandled promise rejections terminate.
process.on('unhandledRejection', e => { throw e });

main();
