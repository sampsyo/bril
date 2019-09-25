#!/usr/bin/env node
import * as bril from './bril';
import {readStdin, unreachable} from './util';

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
  br: 3,
  jmp: 1,
  ret: 0,
  nop: 0,
  access: 2,
  print: null,
};

type Value = boolean | BigInt | Record;
type RecordBindings = {[index: string]: Value};
type Env = Map<bril.Ident, Value>;
type TypeEnv = Map<bril.Ident, bril.RecordType>;

interface Record {
  name: string;
  bindings: RecordBindings;
}

function get<T>(env: Map<bril.Ident, T>, ident: bril.Ident) : T {
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
  return checkIntVal(val, env, index, instr.op)
}

function getBool(instr: bril.Operation, env: Env, index: number) : boolean {
  let val = get(env, instr.args[index]);
  return checkBoolVal(val, env, index, instr.op)
}

function checkBoolVal(val: Value, env: Env, index: number | string, op: string) : boolean {
  if (typeof val !== 'boolean') {
    throw `${op} argument ${index} must be a boolean`;
  }
  return val;
}

function checkIntVal(val: Value, env: Env, index: number | string, op: string) {
  if (typeof val !== 'bigint') {
    throw `${op} argument ${index} must be a number`;
  }
  return val;
}

/**
 * Creates a new record given record bindings and a name;
 * @param o RecordBindings to copy
 * @param name Name of new record
 */
function copy(o: RecordBindings, name: string) {
  var output : Record, v, key;
  output = {name: name, bindings: {}};
  for (key in o) {
      v = o[key];
      if (typeof v === "boolean" || typeof v === 'bigint') {
        output.bindings[key] = v;
      } else {
        output.bindings[key] = copy((v as Record).bindings, (v as Record).name)
      }
  }
  return output;
}

/**
 * Creates a record given. Used for opcodes 'recordinst' and 'with'
 * @param init Optional field initialization for new record.
 *             Used for 'with' syntax.
 */
function createRecord(instr: bril.RecordOperation, env: Env, typeEnv: TypeEnv, init?: Record) : Record {
  let record = get(typeEnv, instr.type);
  let fieldList = instr.fields;
  let rec : Record = {name: instr.type, bindings: {}};
  if (!init) { 
    fieldList = record;
  } else {
    rec = copy(init.bindings, instr.type);
  }
  for (let field in fieldList) {
    let declared_type : bril.Type = record[field];
    var val : Value = get(env, instr.fields[field]);
    if (declared_type === "boolean") {
      val = checkBoolVal(val , env, field, instr.op);
    } else if (declared_type === "int") {
      val = checkIntVal(val, env, field, instr.op);
    } else {
      if ((val as Record).name != declared_type) {
        throw `${instr.op} argument ${field} must be a ${declared_type}`;
      } 
    }
    rec.bindings[field] = val;
  }
  return rec;
}

/**
 * The thing to do after interpreting an instruction: either transfer
 * control to a label, go to the next instruction, or end thefunction.
 */
type Action =
  {"label": bril.Ident} |
  {"next": true} |
  {"end": true};
let NEXT: Action = {"next": true};
let END: Action = {"end": true};

/**
 * Interpret an instruction in a given environment, possibly updating the
 * environment. If the instruction branches to a new label, return that label;
 * otherwise, return "next" to indicate that we should proceed to the next
 * instruction or "end" to terminate the function.
 */
function evalInstr(instr: bril.Instruction, env: Env, typeEnv: TypeEnv): Action {
  // Check that we have the right number of arguments.
  if (!(instr.op === "const" || instr.op === "recordinst" ||
    instr.op === "recorddef" || instr.op === "recordwith")) {
    let count = argCounts[instr.op];
    if (count === undefined) {
      throw "unknown opcode " + instr.op;
    } else if (count !== null) {
      checkArgs(instr, count);
    }
  }
  switch (instr.op) {
  case "const": {
    // Ensure that JSON ints get represented appropriately.
    let value: Value;
    if (typeof instr.value === "number") {
      value = BigInt(instr.value);
    } else {
      value = instr.value;
    }

    env.set(instr.dest, value);
    return NEXT;
  }
  
  case "recorddef": {
    typeEnv.set(instr.recordname, instr.fields);
    return NEXT;
  }

  case "recordinst": {
    let val = createRecord(instr, env, typeEnv);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "recordwith": {
    let src_record = get(env, instr.src) as Record; 
    let val = createRecord(instr, env, typeEnv, src_record);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "id": {
    let val = get(env, instr.args[0]);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "access": {
    let record = get(env, instr.args[0]);
    let val = (record as Record).bindings[instr.args[1]];
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
    return END;
  }

  case "nop": {
    return NEXT;
  }
  }
  unreachable(instr);
  throw `unhandled opcode ${(instr as any).op}`;
}

function evalFunc(func: bril.Function) {
  let env: Env = new Map();
  let typeEnv: TypeEnv = new Map();
  for (let i = 0; i < func.instrs.length; ++i) {
    let line = func.instrs[i];
    if ('op' in line) {
      let action = evalInstr(line, env, typeEnv);

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
        return;
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
