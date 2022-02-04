#!/usr/bin/env node
import * as bril from './bril';
import {readStdin} from './util';

type TypeEnv = Map<bril.Ident, bril.Type>;

interface OpType {
  args: bril.Type[],
  dest?: bril.Type,
}

const OP_TYPES: {[key: string]: OpType} = {
  'add': {'args': ['int', 'int'], 'dest': 'int'}
};

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
  // Check for special cases.
  if (instr.op === "print") {
    if ('type' in instr) {
      console.error(`print should have no result type`);
    }
    return;
  }

  // Do we know this operation?
  let opType = OP_TYPES[instr.op];
  if (!opType) {
    console.error(`unknown opcode ${instr.op}`);
    return;
  }

  // Check the argument count.
  let args: bril.Ident[];
  if ('args' in instr && instr.args !== undefined) {
    args = instr.args;
  } else {
    args = [];
  }
  if (args.length !== opType.args.length) {
    console.error(
      `${instr.op} needs ${opType.args.length} args; found ${args.length}`
    );
  }

  // Check the argument types.
  for (let i = 0; i < args.length; ++i) {
    let argType = env.get(args[i]);
    if (!argType) {
      console.error(`${args[i]} (arg ${i}) undefined`);
      continue;
    }
    if (opType.args[i] !== argType) {
      console.error(
        `${args[i]} has type ${argType}, but arg ${i} should ` +
        `have type ${opType.args[i]}`
      );
    }
  }

  // Check destination type.
  if ('type' in instr) {
    if ('dest' in opType) {
      if (instr.type !== opType.dest) {
        console.error(
          `result type of ${instr.op} should be ${opType.dest}, ` +
          `but found ${instr.type}`
        );
      }
    } else {
      console.error(`${instr.op} should have no result type`);
    }
  } else {
    if ('dest' in opType) {
      console.error(`missing result type ${opType.dest} for ${instr.op}`);
    }
  }
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
