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
  call: null,
  handle: 2,
  throw: 1,
};

type Env = Map<bril.Ident, bril.Value>;
type HandlerEnv = Map<bril.Ident, bril.Label>;
type PCIndex = number;
type StackFrame = [bril.Function, Env, HandlerEnv, PCIndex]; // function name, locals, PC function index
type ProgramStack = Array<StackFrame>;

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

/**
 * The thing to do after interpreting an instruction: either transfer
 * control to a label, go to the next instruction, or end thefunction.
 */
type Action =
  {"label": bril.Ident} |
  {"next": true} |
  {"call": bril.Ident, "args": bril.Ident[]} | // call a function
  {"handle": bril.Ident, "handlerLabel": bril.Label} | // install an exception handler
  {"throw": bril.Ident} | // throw an exception
  {"end": true};
let NEXT: Action = {"next": true};
let END: Action = {"end": true};

/**
 * Get pc index of label in a function.
 */
function getLabelIndex(func: bril.Function, lab: bril.Label): number|undefined {
  for (let i = 0; i < func.instrs.length; i++) {
    let sLine = func.instrs[i];
    if ('label' in sLine && sLine.label === lab.label) {
      return i;
    }
  }

  return undefined;
}

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

  case "call": {
    return {"call": instr.args[0], "args": instr.args.slice(1)};
  }

  case "handle": {
    return {"handle": instr.args[0], "handlerLabel": {label: instr.args[1]}};
  }

  case "throw": {
    return {"throw": instr.args[0]};
  }
  }
  unreachable(instr);
  throw `unhandled opcode ${(instr as any).op}`;
}

function isCorrectType(val: number|boolean, type: bril.Type) {
  return (typeof val === 'number' && type === "int") ||
         (typeof val === 'boolean' && type === "bool");
}

function evalFunc(prog: bril.Program, stack:ProgramStack,
    beginFunc: bril.Function, beginEnv: Env, beginHandlerEnv: HandlerEnv, beginPc: PCIndex)
{
  let func = beginFunc;
  let env = beginEnv;
  let handlerEnv = beginHandlerEnv;
  let pc = beginPc;
  while (pc < func.instrs.length) {
    let line = func.instrs[pc];

    // instruction
    if ('op' in line) {
      let action = evalInstr(line, env);

      if ('next' in action) {
        pc++;

      } else if ('label' in action) {
        // Search for the label and transfer control.
        let labelIndex = getLabelIndex(func, {label: action.label});

        if (labelIndex != undefined) {
          pc = labelIndex;

        } else {
          throw `label ${action.label} not found`;
        }

      } else if ('call' in action) {
        let calleeFound = false;

        for (let callee of prog.functions) {
          if (callee.name === action.call) {
            // push current activation record into stack frame
            stack.push([func, env, handlerEnv, pc+1]);
            func = callee;

            if (func.args.length === action.args.length) {
              let calleeEnv = new Map();
              for (let i = 0; i < func.args.length; i++) {
                let formal = func.args[i];
                let actual = action.args[i];
                let actualVal = env.get(actual);
                if (actualVal != undefined) {
                  if (isCorrectType(actualVal, formal.type)) {
                    calleeEnv.set(formal.ident, actualVal);

                  } else {
                    throw `function ${callee.name} argument ${formal.ident} has type ${formal.type}`;
                  }

                } else {
                  throw `variable ${actual} does not exist`;
                }
              }

              env = calleeEnv;
              handlerEnv = new Map();
              pc = 0;
              calleeFound = true;
              break;

            } else {
              throw `function ${callee.name} expects ${func.args.length} arguments`;
            }
          }
        }

        if (!calleeFound) {
          throw `function ${action.call} not found`;
        }

      } else if ('handle' in action) {
        handlerEnv.set(action.handle, action.handlerLabel);
        pc++;

      } else if ('throw' in action) {
        let handler = handlerEnv.get(action.throw);

        // unwind stack!
        while (handler === undefined) {
          let frame = stack.pop();
          if (frame != undefined) {
            let [frameFunc, frameEnv, frameHandlerEnv, framePc] = frame;
            func = frameFunc;
            env = frameEnv;
            handlerEnv = frameHandlerEnv;
            pc = framePc;
            handler = frameHandlerEnv.get(action.throw);

          } else {
            break;
          }
        }

        if (handler != undefined) {
          // Search for the label and transfer control.
          let handlerLabelIndex = getLabelIndex(func, handler);

          if (handlerLabelIndex != undefined) {
            pc = handlerLabelIndex;

          } else {
            throw `label ${handler.label} not found`;
          }

        } else {
          throw `exception ${action.throw} not handled`;
        }

      } else if ('end' in action) {
        // restore the activation record at the top of the stack
        let oldFrame = stack.pop();
        if (oldFrame != undefined) {
          let [oldFunc, oldEnv, oldHandlerEnv, oldPc] = oldFrame;
          func = oldFunc;
          env = oldEnv;
          handlerEnv = oldHandlerEnv;
          pc = oldPc;

        } else {
          return;
        }
      }

    } else { // label; advance program counter
      pc++;
    }
  }
}

function evalProg(prog: bril.Program) {
  for (let func of prog.functions) {
    if (func.name === "main") {
      evalFunc(prog, [], func, new Map(), new Map(), 0);
    }
  }
}

async function main() {
  try {
    let prog = JSON.parse(await readStdin()) as bril.Program;
    evalProg(prog);
  } catch (e) {
    console.log(e);
  }
}

// Make unhandled promise rejections terminate.
process.on('unhandledRejection', e => { throw e });

main();
