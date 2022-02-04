#!/usr/bin/env node
import * as bril from './bril';
import {readStdin} from './util';

type TypeEnv = Map<bril.Ident, bril.Type>;

/**
 * Set the type of variable `id` to `type` in `env`, checking for conflicts
 * with the old type for the variable.
 */
function addType(env: TypeEnv, id: bril.Ident, type: bril.Type) {
  let oldType = env.get(id);
  if (oldType) {
    if (oldType !== type) {
      console.error(
        `new type ${type} for ${id} conflicts with old type ${oldType}`
      );
    }
  } else {
    env.set(id, type);
  }
}

function checkInstr(
  env: TypeEnv, labels: Set<bril.Ident>, instr: bril.Instruction
) {

}

function checkFunc(func: bril.Function) {
  let env: TypeEnv = new Map();
  let labels = new Set<bril.Ident>();

  // Initilize the type environment with the arguments.
  if (func.args) {
    for (let arg of func.args) {
      addType(env, arg.name, arg.type);
    }
  }

  // Gather up all the types of the local variables and all the label names.
  for (let instr of func.instrs) {
    if ('dest' in instr) {
      addType(env, instr.dest, instr.type);
    } else if ('label' in instr) {
      labels.add(instr.label);
    }
  }

  // Check each instruction.
  for (let instr of func.instrs) {
    if ('op' in instr) {
      checkInstr(env, labels, instr);
    }
  }
}

function checkProg(prog: bril.Program) {
  for (let func of prog.functions) {
    checkFunc(func);
  }
}

async function main() {
  let prog = JSON.parse(await readStdin()) as bril.Program;
  checkProg(prog);
}

// Make unhandled promise rejections terminate.
process.on('unhandledRejection', e => { throw e });

main();
