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
  print: null,  // Any number of arguments.
  br: 3,
  jmp: 1,
  ret: null, // 0 or 1
  nop: 0,
  push: null,  // Any number of arguments.
  popargs: 1,
  call: 1,
  retval: 0, // 0 or 1
};

type Value = boolean | BigInt;
type Env = Map<bril.Ident, Value>;
type PC = number;
type Call = {
    "name": bril.Ident,
    "env": Env,
    "pc": PC
};
// TODO: Need to keep track of PC as well
type CallStack = Array<Call>;

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
  if (typeof val !== 'bigint') {
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
 * The thing to do after interpreting an instruction: either transfer
 * control to a label, go to the next instruction, or end thefunction.
 */
type Action =
  {"call": bril.Ident} |
  {"push": bril.Ident[]} |
  {"label": bril.Ident} |
  {"pop": true} |
  {"next": true} |
  {"end": bril.Ident[]};
let NEXT: Action = {"next": true};

/**
 * Interpret an instruction in a given environment, possibly updating the
 * environment. If the instruction branches to a new label, return that label;
 * otherwise, return "next" to indicate that we should proceed to the next
 * instruction or "end" to terminate the function.
 */
function evalInstr(instr: bril.Instruction, callStack: CallStack): Action {
    let env = callStack[callStack.length - 1].env;
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
        return {"end": instr.args};
    }

    case "nop": {
        return NEXT;
    }

    case "call": {
        return {"call": instr.args[0]};
    }
    case "push": {
        return {"push": instr.args};
    }
    case "popargs": {
        return {"pop": true};
    }
    case "retval": {
        let val = get(env, "RETVAL");
        env.set(instr.dest, val);
        return NEXT;
    }
    }
    unreachable(instr);
    throw `unhandled opcode ${(instr as any).op}`;
}

function getIds(prog: bril.Program, name: bril.Ident) {
    for (let func of prog.functions) {
        if (func.name === name) {
            return func.args.map((arg) => arg.name);
        }
    }
    throw `function ${name} not found`;
}

function evalFunc(prog: bril.Program, func: bril.Function, callStack: CallStack) {
    // Arguments to the next function call
    let nextArgs: bril.Ident[] = [];
    let currCall = callStack[callStack.length-1];
    for (; currCall.pc < func.instrs.length; ++currCall.pc) {
        let line = func.instrs[currCall.pc];
        if ('op' in line) {
            let action = evalInstr(line, callStack);

            if ('label' in action) {
                // Search for the label and transfer control.
                let i;
                for (i = 0; i < func.instrs.length; ++i) {
                    let sLine = func.instrs[i];
                    if ('label' in sLine && sLine.label === action.label) {
                        break;
                    }
                }
                if (i === func.instrs.length) {
                    // The label is a function name, so change the call stack's
                    // function to be this new one, with a PC of 0, and
                    // correctly bind the arguments
                    let fnArgIds = getIds(prog, action.label);
                    for (let i = 0; i < fnArgIds.length; ++i) {
                        callStack[callStack.length-1].env.set(
                            fnArgIds[i], get(currCall.env, nextArgs[i])
                        );
                    }
                    callStack[callStack.length-1].name = action.label;
                    callStack[callStack.length-1].pc = 0;
                    return;
                }
            } else if ('push' in action) {
                nextArgs = action.push;
            } else if ('call' in action) {
                // Get all the function parameter names of the new call
                // and map them to the values found in nextArgs
                let nextEnv = new Map();
                let fnArgIds = getIds(prog, action.call);
                for (let i = 0; i < fnArgIds.length; ++i) {
                    nextEnv.set(fnArgIds[i], get(currCall.env, nextArgs[i]));
                }
                callStack.push({
                    name: action.call,
                    env: nextEnv,
                    pc: 0
                });
                ++currCall.pc;
                return;
            } else if ('end' in action) {
                callStack.pop();
                if (action.end.length != 0 && callStack.length != 0) {
                    let retval = get(currCall.env, action.end[0]);
                    callStack[callStack.length-1].env.set("RETVAL", retval);
                }
                return;
            }
        }
    }
    callStack.pop();
}

function evalProg(prog: bril.Program) {
    let callStack: CallStack = [{name: "main", env: new Map(), pc: 0}];
    while (callStack.length != 0) {
        let next_func = callStack[callStack.length - 1];
        for (let func of prog.functions) {
            if (func.name === next_func.name) {
                evalFunc(prog, func, callStack);
            }
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
