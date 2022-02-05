#!/usr/bin/env node
import * as bril from './bril';
import {readStdin} from './util';

interface FuncType {
  'ret': bril.Type | undefined,
  'args': bril.Type[],
}
type VarEnv = Map<bril.Ident, bril.Type>;
type FuncEnv = Map<bril.Ident, FuncType>;

interface Env {
  vars: VarEnv;
  labels: Set<bril.Ident>;
  funcs: FuncEnv;
}

interface OpType {
  args: bril.Type[],
  dest?: bril.Type,
  labels?: number,
  funcs?: number,
}

const OP_TYPES: {[key: string]: OpType} = {
  'add': {'args': ['int', 'int'], 'dest': 'int'},
  'mul': {'args': ['int', 'int'], 'dest': 'int'},
  'sub': {'args': ['int', 'int'], 'dest': 'int'},
  'div': {'args': ['int', 'int'], 'dest': 'int'},
  'eq': {'args': ['int', 'int'], 'dest': 'bool'},
  'lt': {'args': ['int', 'int'], 'dest': 'bool'},
  'gt': {'args': ['int', 'int'], 'dest': 'bool'},
  'le': {'args': ['int', 'int'], 'dest': 'bool'},
  'ge': {'args': ['int', 'int'], 'dest': 'bool'},
  'not': {'args': ['bool'], 'dest': 'bool'},
  'and': {'args': ['bool', 'bool'], 'dest': 'bool'},
  'or': {'args': ['bool', 'bool'], 'dest': 'bool'},
  'jmp': {'args': [], 'labels': 1},
  'br': {'args': ['bool'], 'labels': 2},
  'fadd': {'args': ['float', 'float'], 'dest': 'float'},
  'fmul': {'args': ['float', 'float'], 'dest': 'float'},
  'fsub': {'args': ['float', 'float'], 'dest': 'float'},
  'fdiv': {'args': ['float', 'float'], 'dest': 'float'},
  'feq': {'args': ['float', 'float'], 'dest': 'bool'},
  'flt': {'args': ['float', 'float'], 'dest': 'bool'},
  'fgt': {'args': ['float', 'float'], 'dest': 'bool'},
  'fle': {'args': ['float', 'float'], 'dest': 'bool'},
  'fge': {'args': ['float', 'float'], 'dest': 'bool'},
};

const CONST_TYPES: {[key: string]: string} = {
  'int': 'number',
  'float': 'number',
  'bool': 'boolean',
};

/**
 * Set the type of variable `id` to `type` in `env`, checking for conflicts
 * with the old type for the variable.
 */
function addType(env: VarEnv, id: bril.Ident, type: bril.Type) {
  let oldType = env.get(id);
  if (oldType) {
    if (!typeEq(oldType, type)) {
      console.error(
        `new type ${type} for ${id} conflicts with old type ${oldType}`
      );
    }
  } else {
    env.set(id, type);
  }
}

/**
 * Check for type equality.
 */
function typeEq(a: bril.Type, b: bril.Type): boolean {
  if (typeof a === "string" && typeof b === "string") {
    return a == b;
  } else if (typeof a === "object" && typeof b === "object") {
    return typeEq(a.ptr, b.ptr);
  } else {
    return false;
  }
}

function checkArgs(env: Env, args: bril.Ident[], params: bril.Type[]) {
  for (let i = 0; i < args.length; ++i) {
    let argType = env.vars.get(args[i]);
    if (!argType) {
      console.error(`${args[i]} (arg ${i}) undefined`);
      continue;
    }
    if (!typeEq(params[i], argType)) {
      console.error(
        `${args[i]} has type ${argType}, but arg ${i} should ` +
        `have type ${params[i]}`
      );
    }
  }
}

function checkInstr(env: Env, instr: bril.Operation, ret: bril.Type | undefined) {
  let args = instr.args ?? [];

  // Check for special cases.
  if (instr.op === "print") {
    if ('type' in instr) {
      console.error(`print should have no result type`);
    }
    return;
  } else if (instr.op === "id") {
    if (args.length !== 1) {
      console.error(`id should have one arg, not ${args.length}`);
      return;
    }
    let argType = env.vars.get(args[0]);
    if (!argType) {
      console.error(`${args[0]} is undefined`);
    } else if (!typeEq(instr.type, argType)) {
      console.error(`id arg type ${argType} does not match type ${instr.type}`);
    }
    return;
  } else if (instr.op == "call") {
    let funcs = instr.funcs ?? [];
    if (funcs.length !== 1) {
      console.error(`call should have one function, not ${funcs.length}`);
      return;
    }

    let funcType = env.funcs.get(funcs[0]);
    if (!funcType) {
      console.error(`function @${funcs[0]} undefined`);
      return;
    }

    if (funcType.ret) {
      if ('type' in instr) {
        if (!typeEq(instr.type, funcType.ret)) {
          console.error(
            `@${funcs[0]} returns type ${funcType.ret}, not ${instr.type}`
          );
        }
      }
    } else if ('type' in instr) {
      console.error(`@${funcs[0]} does not return a value`);
    }
    
    if (args.length !== funcType.args.length) {
      console.error(
        `@${funcs[0]} expects ${funcType.args.length} args, not ${args.length}`
      );
    } else {
      checkArgs(env, args, funcType.args);
    }

    return;
  } else if (instr.op === "ret") {
    if (ret) {
      if (args.length === 0) {
        console.error(`missing return value in function with return type`);
      } else if (args.length !== 1) {
        console.error(`cannot return multiple values`);
      } else {
        checkArgs(env, args, [ret]);
      }
    } else {
      if (args.length !== 0) {
        console.error(`returning value in function without a return type`);
      }
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
  if (args.length !== opType.args.length) {
    console.error(
      `${instr.op} needs ${opType.args.length} args; found ${args.length}`
    );
  }

  // Check the argument types.
  checkArgs(env, args, opType.args);

  // Check destination type.
  if ('type' in instr) {
    if (opType.dest) {
      if (!typeEq(instr.type, opType.dest)) {
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
  
  // Check labels.
  let labs = instr.labels ?? [];
  let labCount = opType.labels ?? 0;
  if (labs.length !== labCount) {
    console.error(`${instr.op} needs ${labCount} labels; found ${labs.length}`);
  } else {
    for (let lab of labs) {
      if (!env.labels.has(lab)) {
        console.error(`label .${lab} undefined`);
      }
    }
  }
}

function checkConst(instr: bril.Constant) {
  if (!('type' in instr)) {
    console.error(`const missing type`);
    return;
  }
  if (typeof instr.type !== 'string') {
    console.error(`const of non-primitive type ${instr.type}`);
    return;
  }

  let valType = CONST_TYPES[instr.type];
  if (!valType) {
    console.error(`unknown const type ${instr.type}`);
    return;
  }

  if (typeof instr.value !== valType) {
    console.error(
      `const value ${instr.value} does not match type ${instr.type}`
    );
  }
}

function checkFunc(funcs: FuncEnv, func: bril.Function) {
  let vars: VarEnv = new Map();
  let labels = new Set<bril.Ident>();

  // Initilize the type environment with the arguments.
  if (func.args) {
    for (let arg of func.args) {
      addType(vars, arg.name, arg.type);
    }
  }

  // Gather up all the types of the local variables and all the label names.
  for (let instr of func.instrs) {
    if ('dest' in instr) {
      addType(vars, instr.dest, instr.type);
    } else if ('label' in instr) {
      if (labels.has(instr.label)) {
        console.error(`multiply defined label .${instr.label}`);
      } else {
        labels.add(instr.label);
      }
    }
  }

  // Check each instruction.
  for (let instr of func.instrs) {
    if ('op' in instr) {
      if (instr.op === 'const') {
        checkConst(instr);
      } else {
        checkInstr({vars, labels, funcs}, instr, func.type);
      }
    }
  }
}

function checkProg(prog: bril.Program) {
  // Gather up function types.
  let funcEnv: FuncEnv = new Map();
  for (let func of prog.functions) {
    funcEnv.set(func.name, {
      'ret': func.type,
      'args': func.args?.map(a => a.type) ?? [],
    });
  }

  // Check each function.
  for (let func of prog.functions) {
    checkFunc(funcEnv, func);
  }
}

async function main() {
  let prog = JSON.parse(await readStdin()) as bril.Program;
  checkProg(prog);
}

// Make unhandled promise rejections terminate.
process.on('unhandledRejection', e => { throw e });

main();
