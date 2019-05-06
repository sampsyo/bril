import * as bril from './bril';
import {readStdin, unreachable} from './util';

type Env = Map<bril.Ident, bril.Value>;

function get(env: Env, ident: bril.Ident) {
  let val = env.get(ident);
  if (!val) {
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

function evalInstr(instr: bril.Instruction, env: Env) {
  switch (instr.op) {
  case "const":
    env.set(instr.dest, instr.value);
    break;

  case bril.OpCode.id: {
    checkArgs(instr, 1);
    let val = get(env, instr.args[0]);
    env.set(instr.dest, val);
    break;
  }

  case bril.OpCode.add: {
    checkArgs(instr, 2);
    let val = getInt(instr, env, 0) + getInt(instr, env, 1);
    env.set(instr.dest, val);
    break;
  }

  case bril.OpCode.le: {
    checkArgs(instr, 2);
    let val = getInt(instr, env, 0) <= getInt(instr, env, 1);
    env.set(instr.dest, val);
    break;
  }

  case bril.OpCode.lt: {
    checkArgs(instr, 2);
    let val = getInt(instr, env, 0) < getInt(instr, env, 1);
    env.set(instr.dest, val);
    break;
  }

  case bril.OpCode.gt: {
    checkArgs(instr, 2);
    let val = getInt(instr, env, 0) > getInt(instr, env, 1);
    env.set(instr.dest, val);
    break;
  }

  case bril.OpCode.ge: {
    checkArgs(instr, 2);
    let val = getInt(instr, env, 0) >= getInt(instr, env, 1);
    env.set(instr.dest, val);
    break;
  }

  case bril.OpCode.eq: {
    checkArgs(instr, 2);
    let val = getInt(instr, env, 0) === getInt(instr, env, 1);
    env.set(instr.dest, val);
    break;
  }

  case bril.OpCode.not: {
    checkArgs(instr, 1);
    let val = !getBool(instr, env, 0);
    env.set(instr.dest, val);
    break;
  }

  case bril.OpCode.and: {
    checkArgs(instr, 2);
    let val = getBool(instr, env, 0) && getBool(instr, env, 1);
    env.set(instr.dest, val);
    break;
  }

  case bril.OpCode.or: {
    checkArgs(instr, 2);
    let val = getBool(instr, env, 0) || getBool(instr, env, 1);
    env.set(instr.dest, val);
    break;
  }

  case bril.OpCode.print: {
    let values = instr.args.map(i => get(env, i));
    console.log(...values);
    break;
  }

  default:
    unreachable(instr);
  }
}

function evalFunc(func: bril.Function) {
  let env: Env = new Map();
  for (let instr of func.instrs) {
    evalInstr(instr, env);
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
