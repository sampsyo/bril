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
  not: 2,
  and: 2,
  or: 2,
  print: null,  // Any number of arguments.
  br: 3,
  jmp: 1,
  ret: 0,
  nop: 0,
  lw: 1,
  sw: 2,
  vadd: 2,
  vload: 1,
  vstore: 2,
  _vadd: 2,
  _vload: 1,
  _vstore: 2
};

// this represents an infinite size register file
type Env = Map<bril.Ident, bril.Value>;

/*
 * Declare an array of memory to represent a stack-like memory structure.
 * Locations of memory dictated by software and freed if ever change stack frame/function
 * The freeing isn't supported yet because there is only one function
 */
let stackSize: number = 1024;
let stack = new Array<number>(stackSize);

/*
 * We're doing fixed array size of 4 so set this here
 * It's all my computer support natively so don't go beyond
 */
let fixedVecSize: number = 4;

/*
 * Initialize the binding
 */

//let nbind = require('nbind');
//let lib = nbind.init().lib;

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

// memory lookup
function getMem(addr: number) {
  if (addr < stackSize) {
    let val = stack[addr];
    return val;
  }
  else {
    throw `load with addr ${addr} out of range of stack`;
  }
}

// memory write
function setMem(val: number, addr: number) {
  if (addr < stackSize) {
    stack[addr] = val;
  }
  else {
    throw `store addr ${addr} out of range of stack`;
  }
}

// get vector value from vector register file
function getVec(instr: bril.Operation, env: Env, index: number) {
  let val = get(env, instr.args[index]);
  if (!(val instanceof Int32Array)) {
    throw `${instr.op} argument ${index} must be a Int32Array`;
  }

  return val;
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
function evalInstr(instr: bril.Instruction, env: Env): Action {
  // Check that we have the right number of arguments.
  if (instr.op !== "const") {
    let count = argCounts[instr.op];
    if (count === undefined) {
      throw "unknown opcode " + instr.op;
    } else if (count !== null) {
      checkArgs(instr, count);
    }
  }

  switch (instr.op) {
  case "const":
    env.set(instr.dest, instr.value);
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
    let values = instr.args.map(i => get(env, i));
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


  case "lw": {
    // lookup memory based on value in register
    let addr = getInt(instr, env, 0);
    let val = getMem(addr);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "sw": {
    let val = getInt(instr, env, 0);
    let addr = getInt(instr, env, 1);
    setMem(val, addr);
    return NEXT;
  }

  case "vadd": {

    // serialized version
    let vecA = getVec(instr, env, 0);
    let vecB = getVec(instr, env, 1);
    let vecC = new Int32Array(fixedVecSize);
    for (let i = 0; i < fixedVecSize; i++) {
      vecC[i] = vecA[i] + vecB[i];
    }
    env.set(instr.dest, vecC);    

    //console.log(c);
    return NEXT;
  }

  case "vload": {

    // serialized version
    let addr = getInt(instr, env, 0);
    let vec = new Int32Array(4);
    for (let i = 0; i < fixedVecSize; i++) {
      vec[i] = getMem(addr + i);
    }
    env.set(instr.dest, vec);
    return NEXT;
  }

  case "vstore": {
    
    // serialized version
    let val = getVec(instr, env, 0);
    let addr = getInt(instr, env, 1);
    for (let i = 0; i < fixedVecSize; i++) {
      setMem(val[i], addr + i);
    }

    return NEXT;
  }

  case "_vadd": {
    /*let nbind = require('nbind');
let lib = nbind.init().lib;
    // if end up doing big vectors, then need to compare serial c impl as well
    let vecA = getVec(instr, env, 0);
    let vecB = getVec(instr, env, 1);
    //let vecA = new Int32Array(fixedVecSize);
    //let vecB = new Int32Array(fixedVecSize);
    let vecC = new Array<number>(fixedVecSize);
    //let vecC = new Int32Array(fixedVecSize);
    lib.SIMD.vecAdd(vecA, vecB, vecC);
    env.set(instr.dest, vecC);
    console.log("_vadd");*/

    return NEXT;
  }

  case "_vload": {

    // serialized version
    /*let addr = getInt(instr, env, 0);
    let vec = new Array<number>(4);
    for (let i = 0; i < fixedVecSize; i++) {
      vec[i] = getMem(addr + i);
    }
    env.set(instr.dest, vec);*/
    return NEXT;
  }

  case "_vstore": {
    
    // serialized version
    /*let val = getVec(instr, env, 0);
    let addr = getInt(instr, env, 1);
    for (let i = 0; i < fixedVecSize; i++) {
      setMem(val[i], addr + i);
    }*/

    return NEXT;
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
      let action = evalInstr(line, env);

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
