import * as bril from './bril';
import {readStdin, unreachable} from './util';

type Env = Map<bril.Ident, bril.ConstValue>;

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
    let lhs = get(env, instr.args[0]);
    let rhs = get(env, instr.args[1]);
    let val = lhs + rhs;
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
