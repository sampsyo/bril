#!/usr/bin/env node
import * as bril from './bril';
import {readStdin, unreachable} from './util';

const argCounts:{[key in bril.OpCode]: number | null} = {
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
  not: 2,
  and: 2,
  or: 2,
  print: null,  // Any number of arguments.
  br: 3,
  jmp: 1,
};

type Env = Map<bril.Ident, bril.Value>;

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
  if (typeof val !== 'number') {
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
 * Interpret an instruction in a given environment, possibly updating the
 * environment. If the instruction branches to a new label, return that label;
 * otherwise, return `null` to indicate that we should proceed to the next
 * instruction.
 */
function evalInstr(instr: bril.Instruction, env: Env): bril.Ident | null {
  // Check that we have the right number of arguments.
  if (instr.op !== "const") {
    let count = argCounts[instr.op];
    if (count !== null) {
      checkArgs(instr, count);
    }
  }

  switch (instr.op) {
  case "const":
    env.set(instr.dest, instr.value);
    return null;

  case "id": {
    let val = get(env, instr.args[0]);
    env.set(instr.dest, val);
    return null;
  }

  case "add": {
    let val = getInt(instr, env, 0) + getInt(instr, env, 1);
    env.set(instr.dest, val);
    return null;
  }

  case "mul": {
    let val = getInt(instr, env, 0) * getInt(instr, env, 1);
    env.set(instr.dest, val);
    return null;
  }

  case "sub": {
    let val = getInt(instr, env, 0) - getInt(instr, env, 1);
    env.set(instr.dest, val);
    return null;
  }

  case "div": {
    let val = getInt(instr, env, 0) / getInt(instr, env, 1);
    env.set(instr.dest, val);
    return null;
  }

  case "le": {
    let val = getInt(instr, env, 0) <= getInt(instr, env, 1);
    env.set(instr.dest, val);
    return null;
  }

  case "lt": {
    let val = getInt(instr, env, 0) < getInt(instr, env, 1);
    env.set(instr.dest, val);
    return null;
  }

  case "gt": {
    let val = getInt(instr, env, 0) > getInt(instr, env, 1);
    env.set(instr.dest, val);
    return null;
  }

  case "ge": {
    let val = getInt(instr, env, 0) >= getInt(instr, env, 1);
    env.set(instr.dest, val);
    return null;
  }

  case "eq": {
    let val = getInt(instr, env, 0) === getInt(instr, env, 1);
    env.set(instr.dest, val);
    return null;
  }

  case "not": {
    let val = !getBool(instr, env, 0);
    env.set(instr.dest, val);
    return null;
  }

  case "and": {
    let val = getBool(instr, env, 0) && getBool(instr, env, 1);
    env.set(instr.dest, val);
    return null;
  }

  case "or": {
    let val = getBool(instr, env, 0) || getBool(instr, env, 1);
    env.set(instr.dest, val);
    return null;
  }

  case "print": {
    let values = instr.args.map(i => get(env, i));
    console.log(...values);
    return null;
  }

  case "jmp": {
    return instr.args[0];
  }

  case "br": {
    let cond = getBool(instr, env, 0);
    if (cond) {
      return instr.args[1];
    } else {
      return instr.args[2];
    }
  }
  }
  unreachable(instr);
  throw `unhandled opcode ${(instr as any).op}`;
}

function evalFunc(func: bril.Function) {
  let env: Env = new Map();
  for (let i = 0; i < func.instrs.length; ++i) {
    let line = func.instrs[i];
    if ('op' in line) {
      let destLabel = evalInstr(line, env);

      // Search for the label and transfer control.
      if (destLabel) {
        for (i = 0; i < func.instrs.length; ++i) {
          let sLine = func.instrs[i];
          if ('label' in sLine && sLine.label === destLabel) {
            break;
          }
        }
        if (i === func.instrs.length) {
          throw `label ${destLabel} not found`;
        }
      }
    }
  }
}

function evalProg(prog: bril.Program) {
  for (let func of prog.functions) {
    if (func.name === "main") {
      evalFunc(func);
    }
  }
}

async function main() {
  let prog = JSON.parse(await readStdin()) as bril.Program;
  evalProg(prog);
}

// Make unhandled promise rejections terminate.
process.on('unhandledRejection', e => { throw e });

main();
