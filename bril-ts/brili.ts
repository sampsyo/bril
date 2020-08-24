#!/usr/bin/env node
import * as bril from './bril';
import {readStdin, unreachable} from './util';
import { BigIntLiteral } from 'typescript';

/**
 * An interpreter error to print to the console.
 */
class BriliError extends Error {
  constructor(message?: string) {
    super(message);
    Object.setPrototypeOf(this, new.target.prototype);
    this.name = BriliError.name;
  }
}

/**
 * Create an interpreter error object to throw.
 */
function error(message: string): BriliError {
  return new BriliError(message);
}

/**
 * An abstract key class used to access the heap.
 * This allows for "pointer arithmetic" on keys,
 * while still allowing lookups based on the based pointer of each allocation.
 */
export class Key {
    readonly base: number;
    readonly offset: number;

    constructor(b:number, o:number) {
        this.base = b;
        this.offset = o;
    }

    add(offset:number) {
        return new Key(this.base, this.offset + offset);
    }
}

/**
 * A Heap maps Keys to arrays of a given type.
 */
export class Heap<X> {

    private readonly storage: Map<number, X[]>
    constructor() {
        this.storage = new Map()
    }

    isEmpty(): boolean {
        return this.storage.size == 0;
    }

    private count = 0;
    private getNewBase():number {
        let val = this.count;
        this.count++;
        return val;
    }

    private freeKey(key:Key) {
        return;
    }

    alloc(amt:number): Key {
        if (amt <= 0) {
            throw error(`cannot allocate ${amt} entries`);
        }
        let base = this.getNewBase();
        this.storage.set(base, new Array(amt))
        return new Key(base, 0);
    }

    free(key: Key) {
        if (this.storage.has(key.base) && key.offset == 0) {
            this.freeKey(key);
            this.storage.delete(key.base);
        } else {
            throw error(`Tried to free illegal memory location base: ${key.base}, offset: ${key.offset}. Offset must be 0.`);
        }
    }

    write(key: Key, val: X) {
        let data = this.storage.get(key.base);
        if (data && data.length > key.offset && key.offset >= 0) {
            data[key.offset] = val;
        } else {
            throw error(`Uninitialized heap location ${key.base} and/or illegal offset ${key.offset}`);
        }
    }

    read(key: Key): X {
        let data = this.storage.get(key.base);
        if (data && data.length > key.offset && key.offset >= 0) {
            return data[key.offset];
        } else {
            throw error(`Uninitialized heap location ${key.base} and/or illegal offset ${key.offset}`);
        }
    }
}

const argCounts: {[key in bril.OpCode]: number | null} = {
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
  ret: null,  // (Should be 0 or 1.)
  nop: 0,
  call: null,
  alloc: 1,
  free: 1,
  store: 2,
  load: 1,
  ptradd: 2,
};

type Pointer = {
  loc: Key;
  type: bril.Type;
}

type Value = boolean | Pointer | BigInt;
type ReturnValue = Value | null;
type Env = Map<bril.Ident, Value>;

/**
 * Check whether a run-time value matches the given static type.
 */
function typeCheck(val: Value, typ: bril.Type): boolean {
  if (typ === "int") {
    return typeof val === "bigint";
  } else if (typ === "bool") {
    return typeof val === "boolean";
  } else if ("ptr" in typ) {
    return val instanceof Key;
  }
  throw error(`unknown type ${typ}`);
}

function get(env: Env, ident: bril.Ident) {
  let val = env.get(ident);
  if (typeof val === 'undefined') {
    throw error(`undefined variable ${ident}`);
  }
  return val;
}

function findFunc(func: bril.Ident, funcs: bril.Function[]) {
  let matches = funcs.filter(function (f: bril.Function) {
    return f.name === func;
  });

  if (matches.length == 0) {
    throw error(`no function of name ${func} found`);
  } else if (matches.length > 1) {
    throw error(`multiple functions of name ${func} found`);
  }

  return matches[0];
}

function alloc(ptrType: bril.PointerType, amt:number, heap:Heap<Value>): Pointer {
  if (typeof ptrType != 'object') {
    throw `unspecified pointer type ${ptrType}`
  } else if (amt <= 0) {
    throw `must allocate a positive amount of memory: ${amt} <= 0`
  } else {
    let loc = heap.alloc(amt)
    let dataType = ptrType.ptr;
    return {
      loc: loc,
      type: dataType
    }
  }
}

/**
 * Ensure that the instruction has exactly `count` arguments,
 * throw an exception otherwise.
 */
function checkArgs(instr: bril.Operation, count: number) {
  if (instr.args.length != count) {
    throw error(`${instr.op} takes ${count} argument(s); got ${instr.args.length}`);
  }
}

function getPtr(instr: bril.Operation, env: Env, index: number): Pointer {
  let val = get(env, instr.args[index]);
  if (typeof val !== 'object' || val instanceof BigInt) {
    throw `${instr.op} argument ${index} must be a Pointer`;
  }
  return val;
}

function getArgument(instr: bril.Operation, env: Env, index: number, 
  typ : bril.Type) {
  let val = get(env, instr.args[index]);
  if (!typeCheck(val, typ)) {
    throw error(`${instr.op} argument ${index} must be a {brilTyp}`);
  }
  return val;
}

function getInt(instr: bril.Operation, env: Env, index: number) : bigint {
  return getArgument(instr, env, index, 'int') as bigint;
}

function getBool(instr: bril.Operation, env: Env, index: number) : boolean {
  return getArgument(instr, env, index, 'bool') as boolean;
}

/**
 * The thing to do after interpreting an instruction: either transfer
 * control to a label, go to the next instruction, or end thefunction.
 */
type Action =
  {"label": bril.Ident} |
  {"next": true} |
  {"end": ReturnValue};
let NEXT: Action = {"next": true};

/**
 * Interpet a call instruction.
 */
function evalCall(instr: bril.Operation, env: Env, heap: Heap<Value>, funcs: bril.Function[])
  : Action {
  let funcName = instr.args[0];
  let funcArgs = instr.args.slice(1);

  let func = findFunc(funcName, funcs);
  if (func === null) {
    throw error(`undefined function ${funcName}`);
  }

  let newEnv: Env = new Map();

  // Check arity of arguments and definition.
  if (func.args.length !== funcArgs.length) {
    throw error(`function expected ${func.args.length} arguments, got ${funcArgs.length}`);
  }

  for (let i = 0; i < func.args.length; i++) {
    // Look up the variable in the current (calling) environment.
    let value = get(env, funcArgs[i]);

    // Check argument types
    if (!typeCheck(value, func.args[i].type)) {
      throw error(`function argument type mismatch`);
    }

    // Set the value of the arg in the new (function) environemt.
    newEnv.set(func.args[i].name, value);
  }

  // Dynamically check the function's return value and type
  let retVal = evalFunc(func, funcs, heap, newEnv);
  if (!('dest' in instr)) {  // `instr` is an `EffectOperation`.
     // Expected void function
    if (retVal !== null) {
      throw error(`unexpected value returned without destination`);
    }
    if (func.type !== undefined) {
      throw error(`non-void function (type: ${func.type}) doesn't return anything`); 
    }
  } else {  // `instr` is a `ValueOperation`.
    // Expected non-void function
    if (instr.type === undefined) {
      throw error(`function call must include a type if it has a destination`);  
    }
    if (instr.dest === undefined) {
      throw error(`function call must include a destination if it has a type`);  
    }
    if (retVal === null) {
      throw error(`non-void function (type: ${func.type}) doesn't return anything`);
    }
    if (!typeCheck(retVal, instr.type)) {
      throw error(`type of value returned by function does not match destination type`);
    }
    if (func.type !== instr.type ) {
      throw error(`type of value returned by function does not match declaration`);
    }
    env.set(instr.dest, retVal);
  }
  return NEXT;
}

/**
 * Interpret an instruction in a given environment, possibly updating the
 * environment. If the instruction branches to a new label, return that label;
 * otherwise, return "next" to indicate that we should proceed to the next
 * instruction or "end" to terminate the function.
 */
function evalInstr(instr: bril.Instruction, env: Env, heap:Heap<Value>, funcs: bril.Function[]): Action {
  // Check that we have the right number of arguments.
  if (instr.op !== "const") {
    let count = argCounts[instr.op];
    if (count === undefined) {
      throw error("unknown opcode " + instr.op);
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
    return NEXT;

  case "id": {
    let val = get(env, instr.args[0]);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "add": {
    let val = getInt(instr, env, 0) + getInt(instr, env, 1);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "mul": {
    let val = getInt(instr, env, 0) * getInt(instr, env, 1);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "sub": {
    let val = getInt(instr, env, 0) - getInt(instr, env, 1);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "div": {
    let val = getInt(instr, env, 0) / getInt(instr, env, 1);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "le": {
    let val = getInt(instr, env, 0) <= getInt(instr, env, 1);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "lt": {
    let val = getInt(instr, env, 0) < getInt(instr, env, 1);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "gt": {
    let val = getInt(instr, env, 0) > getInt(instr, env, 1);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "ge": {
    let val = getInt(instr, env, 0) >= getInt(instr, env, 1);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "eq": {
    let val = getInt(instr, env, 0) === getInt(instr, env, 1);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "not": {
    let val = !getBool(instr, env, 0);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "and": {
    let val = getBool(instr, env, 0) && getBool(instr, env, 1);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "or": {
    let val = getBool(instr, env, 0) || getBool(instr, env, 1);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "print": {
    let values = instr.args.map(i => get(env, i).toString());
    console.log(...values);
    return NEXT;
  }

  case "jmp": {
    return {"label": instr.args[0]};
  }

  case "br": {
    let cond = getBool(instr, env, 0);
    if (cond) {
      return {"label": instr.args[1]};
    } else {
      return {"label": instr.args[2]};
    }
  }
  
  case "ret": {
    let argCount = instr.args.length;
    if (argCount == 0) {
      return {"end": null};
    } else if (argCount == 1) {
      let val = get(env, instr.args[0]);
      return {"end": val};
    } else {
      throw error(`ret takes 0 or 1 argument(s); got ${argCount}`);
    }
  }

  case "nop": {
    return NEXT;
  }

  case "call": {
    return evalCall(instr, env, heap, funcs);
  }

  case "alloc": {
    let amt = getInt(instr, env, 0);
    let typ = instr.type;
    if (!(typeof typ === "object" && typ.hasOwnProperty('ptr'))) {
      throw error(`cannot allocate non-pointer type ${instr.type}`);
    }
    let ptr = alloc(typ, Number(amt), heap);
    env.set(instr.dest, ptr);
    return NEXT;
  }

  case "free": {
    let val = getPtr(instr, env, 0)
    heap.free(val.loc);
    return NEXT;
  }

  case "store": {
    let target = getPtr(instr, env, 0)
    switch (target.type) {
      case "int": {
        heap.write(target.loc, getInt(instr, env, 1))
        break;
      }
      case "bool": {
        heap.write(target.loc, getBool(instr, env, 1))
        break;
      }
      default: {
        heap.write(target.loc, getPtr(instr, env, 1))
        break;
      }
    }
    return NEXT;
  }

  case "load": {
    let ptr = getPtr(instr, env, 0)
    let val = heap.read(ptr.loc)
    if (val == undefined || val == null) {
      throw error(`Pointer ${instr.args[0]} points to uninitialized data`);
    } else {
      env.set(instr.dest, val)
    }
    return NEXT;
  }

  case "ptradd": {
    let ptr = getPtr(instr, env, 0)
    let val = getInt(instr, env, 1)
    env.set(instr.dest, { loc: ptr.loc.add(Number(val)), type: ptr.type })
    return NEXT;
  }
  }
  unreachable(instr);
  throw error(`unhandled opcode ${(instr as any).op}`);
}

function evalFunc(func: bril.Function, funcs: bril.Function[], heap: Heap<Value>, env: Env)
  : ReturnValue {
  for (let i = 0; i < func.instrs.length; ++i) {
    let line = func.instrs[i];
    if ('op' in line) {
      let action = evalInstr(line, env, heap, funcs);

      if ('label' in action) {
        // Search for the label and transfer control.
        for (i = 0; i < func.instrs.length; ++i) {
          let sLine = func.instrs[i];
          if ('label' in sLine && sLine.label === action.label) {
            break;
          }
        }
        if (i === func.instrs.length) {
          throw error(`label ${action.label} not found`);
        }
      } else if ('end' in action) {
        return action.end;
      }
    }
  }

  // Reached the end of the function without hitting `ret`.
  return null;
}

function parseBool(s : string) : boolean {
  if (s === 'true') {
    return true;
  } else if (s === 'false') {
    return false;
  } else {
    throw error(`boolean argument to main must be 'true'/'false'; got ${s}`);
  }
}

function parseMainArguments(expected: bril.Argument[], args: string[]) : Env {
  let newEnv: Env = new Map();

  if (args.length !== expected.length) {
    throw error(`mismatched main argument arity: expected ${expected.length}; got ${args.length}`);
  }

  for (let i = 0; i < args.length; i++) {
    let type = expected[i].type;
    switch (type) {
      case "int":
        let n : bigint = BigInt(parseInt(args[i]));
        newEnv.set(expected[i].name, n as Value);
        break;
      case "bool":
        let b : boolean = parseBool(args[i]);
        newEnv.set(expected[i].name, b as Value);
        break;
    }
  }
  return newEnv;
}

function evalProg(prog: bril.Program) {
  let heap = new Heap<Value>()
  let main = findFunc("main", prog.functions);
  if (main === null) {
    console.log(`warning: no main function defined, doing nothing`);
  } else {
    let expected = main.args;
    let args : string[] = process.argv.slice(2, process.argv.length);
    let newEnv = parseMainArguments(expected, args);
    evalFunc(main, prog.functions, heap, newEnv);
  }
  if (!heap.isEmpty()) {
    throw error(`Some memory locations have not been freed by end of execution.`);
  }
}

async function main() {
  try {
    let prog = JSON.parse(await readStdin()) as bril.Program;
    evalProg(prog);
  }
  catch(e) {
    if (e instanceof BriliError) {
      console.error(`error: ${e.message}`) 
      process.exit(2);
    } else {
      throw e;
    }
  }
}

// Make unhandled promise rejections terminate.
process.on('unhandledRejection', e => { throw e });

main();
