#!/usr/bin/env node
import * as bril from './bril';
import {readStdin, unreachable} from './util';

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

/**
 * Format a type as a human-readable string.
 */
function typeFmt(t: bril.Type): string {
  if (typeof t === "string") {
    return t;
  } else if (typeof t === "object") {
    return `ptr<${typeFmt(t.ptr)}>`;
  }
  unreachable(t);
}

/**
 * Check that an argument list matches a parameter type list.
 */
function checkArgs(env: Env, args: bril.Ident[], params: bril.Type[], name: string) {
  // Check argument count.
  if (args.length !== params.length) {
    console.error(
      `${name} expects ${params.length} args, not ${args.length}`
    );
    return;
  }

  // Check each argument.
  let argc = Math.min(args.length, params.length);
  for (let i = 0; i < argc; ++i) {
    let argType = env.vars.get(args[i]);
    if (!argType) {
      console.error(`${args[i]} (arg ${i}) undefined`);
      continue;
    }
    if (!typeEq(params[i], argType)) {
      console.error(
        `${args[i]} has type ${typeFmt(argType)}, but arg ${i} for ${name} ` +
        `should have type ${typeFmt(params[i])}`
      );
    }
  }
}

/**
 * Check an instruction's arguments and labels against a type specification.
 */
function checkTypes(env: Env, instr: bril.Operation, type: OpType, name?: string) {
  let args = instr.args ?? [];
  name = name ?? instr.op;

  // Check the argument types.
  checkArgs(env, args, type.args, name);

  // Check destination type.
  if ('type' in instr) {
    if (type.dest) {
      if (!typeEq(instr.type, type.dest)) {
        console.error(
          `result type of ${name} should be ${typeFmt(type.dest)}, ` +
          `but found ${typeFmt(instr.type)}`
        );
      }
    } else {
      console.error(`${name} should have no result type`);
    }
  } else {
    if (type.dest) {
      console.error(
        `missing result type ${typeFmt(type.dest)} for ${name}`
      );
    }
  }

  // Check labels.
  let labs = instr.labels ?? [];
  let labCount = type.labels ?? 0;
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

function checkInstr(env: Env, instr: bril.Operation, ret: bril.Type | undefined) {
  let args = instr.args ?? [];

  // Check for special cases.
  if (instr.op === "print") {
    if ('type' in instr) {
      console.error(`print should have no result type`);
    }
    return;
  } else if (instr.op === "id") {
    if (!instr.type) {
      console.error(`missing result type for id`);
    } else {
      checkTypes(env, instr, {
        'args': [instr.type],
        'dest': instr.type,
      });
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
    
    checkTypes(env, instr, {
      'args': funcType.args,
      'dest': funcType.ret,
    }, `@${funcs[0]}`);
    return;
  } else if (instr.op === "ret") {
    if (ret) {
      if (args.length === 0) {
        console.error(`missing return value in function with return type`);
      } else if (args.length !== 1) {
        console.error(`cannot return multiple values`);
      } else {
        checkArgs(env, args, [ret], 'ret');
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

  checkTypes(env, instr, opType);
}

function checkConst(instr: bril.Constant) {
  if (!('type' in instr)) {
    console.error(`const missing type`);
    return;
  }
  if (typeof instr.type !== 'string') {
    console.error(`const of non-primitive type ${typeFmt(instr.type)}`);
    return;
  }

  let valType = CONST_TYPES[instr.type];
  if (!valType) {
    console.error(`unknown const type ${typeFmt(instr.type)}`);
    return;
  }

  if (typeof instr.value !== valType) {
    console.error(
      `const value ${instr.value} does not match type ${typeFmt(instr.type)}`
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

main();
