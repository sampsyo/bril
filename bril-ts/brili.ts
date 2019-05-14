#!/usr/bin/env node
import * as bril from './bril';
import {readStdin, unreachable} from './util';

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
  switch (instr.op) {
  case "const":
    env.set(instr.dest, instr.value);
    return null;

  case bril.OpCode.id: {
    checkArgs(instr, 1);
    let val = get(env, instr.args[0]);
    env.set(instr.dest, val);
    return null;
  }

  case bril.OpCode.add: {
    checkArgs(instr, 2);
    let val = getInt(instr, env, 0) + getInt(instr, env, 1);
    env.set(instr.dest, val);
    return null;
  }

  case bril.OpCode.le: {
    checkArgs(instr, 2);
    let val = getInt(instr, env, 0) <= getInt(instr, env, 1);
    env.set(instr.dest, val);
    return null;
  }

  case bril.OpCode.lt: {
    checkArgs(instr, 2);
    let val = getInt(instr, env, 0) < getInt(instr, env, 1);
    env.set(instr.dest, val);
    return null;
  }

  case bril.OpCode.gt: {
    checkArgs(instr, 2);
    let val = getInt(instr, env, 0) > getInt(instr, env, 1);
    env.set(instr.dest, val);
    return null;
  }

  case bril.OpCode.ge: {
    checkArgs(instr, 2);
    let val = getInt(instr, env, 0) >= getInt(instr, env, 1);
    env.set(instr.dest, val);
    return null;
  }

  case bril.OpCode.eq: {
    checkArgs(instr, 2);
    let val = getInt(instr, env, 0) === getInt(instr, env, 1);
    env.set(instr.dest, val);
    return null;
  }

  case bril.OpCode.not: {
    checkArgs(instr, 1);
    let val = !getBool(instr, env, 0);
    env.set(instr.dest, val);
    return null;
  }

  case bril.OpCode.and: {
    checkArgs(instr, 2);
    let val = getBool(instr, env, 0) && getBool(instr, env, 1);
    env.set(instr.dest, val);
    return null;
  }

  case bril.OpCode.or: {
    checkArgs(instr, 2);
    let val = getBool(instr, env, 0) || getBool(instr, env, 1);
    env.set(instr.dest, val);
    return null;
  }

  case bril.OpCode.print: {
    let values = instr.args.map(i => get(env, i));
    console.log(...values);
    return null;
  }

  case bril.OpCode.jmp: {
    checkArgs(instr, 1);
    return instr.args[0];
  }

  case bril.OpCode.br: {
    checkArgs(instr, 3);
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

main();
