import * as bril from "./bril-ts/bril.ts";
import { readStdin, unreachable } from "./bril-ts/util.ts";

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

  constructor(b: number, o: number) {
    this.base = b;
    this.offset = o;
  }

  add(offset: number) {
    return new Key(this.base, this.offset + offset);
  }
}

/**
 * A Heap maps Keys to arrays of a given type.
 */
export class Heap<X> {
  private readonly storage: Map<number, X[]>;
  constructor() {
    this.storage = new Map();
  }

  isEmpty(): boolean {
    return this.storage.size == 0;
  }

  private count = 0;
  private getNewBase(): number {
    const val = this.count;
    this.count++;
    return val;
  }

  private freeKey(_key: Key) {
    return;
  }

  alloc(amt: number): Key {
    if (amt <= 0) {
      throw error(`cannot allocate ${amt} entries`);
    }
    const base = this.getNewBase();
    this.storage.set(base, new Array(amt));
    return new Key(base, 0);
  }

  free(key: Key) {
    if (this.storage.has(key.base) && key.offset == 0) {
      this.freeKey(key);
      this.storage.delete(key.base);
    } else {
      throw error(
        `Tried to free illegal memory location base: ${key.base}, offset: ${key.offset}. Offset must be 0.`,
      );
    }
  }

  write(key: Key, val: X) {
    const data = this.storage.get(key.base);
    if (data && data.length > key.offset && key.offset >= 0) {
      data[key.offset] = val;
    } else {
      throw error(
        `Uninitialized heap location ${key.base} and/or illegal offset ${key.offset}`,
      );
    }
  }

  read(key: Key): X {
    const data = this.storage.get(key.base);
    if (data && data.length > key.offset && key.offset >= 0) {
      return data[key.offset];
    } else {
      throw error(
        `Uninitialized heap location ${key.base} and/or illegal offset ${key.offset}`,
      );
    }
  }
}

const argCounts: { [key in bril.OpCode]: number | null } = {
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
  fadd: 2,
  fmul: 2,
  fsub: 2,
  fdiv: 2,
  flt: 2,
  fle: 2,
  fgt: 2,
  fge: 2,
  feq: 2,
  print: null, // Any number of arguments.
  br: 1,
  jmp: 0,
  loop: 0, // ADDED
  block: 0, // ADDED
  if: 1, // ADDED
  break: 0, // ADDED
  continue: 0, // ADDED
  ret: null, // (Should be 0 or 1.)
  nop: 0,
  call: null,
  alloc: 1,
  free: 1,
  store: 2,
  load: 1,
  ptradd: 2,
  get: 0,
  set: 2,
  undef: 0,
  speculate: 0,
  guard: 1,
  commit: 0,
  ceq: 2,
  clt: 2,
  cle: 2,
  cgt: 2,
  cge: 2,
  char2int: 1,
  int2char: 1,
};

type Pointer = {
  loc: Key;
  type: bril.Type;
};

const UNDEF = Symbol("undef");

type Value = boolean | bigint | Pointer | number | string | typeof UNDEF;
type Env = Map<bril.Ident, Value>;

/**
 * Check whether a run-time value matches the given static type.
 */
function typeCheck(val: Value, typ: bril.Type): boolean {
  if (typ === "int") {
    return typeof val === "bigint";
  } else if (typ === "bool") {
    return typeof val === "boolean";
  } else if (typ === "float") {
    return typeof val === "number";
  } else if (typeof typ === "object" && Object.hasOwn(typ, "ptr")) {
    return Object.hasOwnProperty.call(val, "loc");
  } else if (typ === "char") {
    return typeof val === "string";
  } else if (typ === "any") {
    return true;
  }
  throw error(`unknown type ${typ}`);
}

/**
 * Check whether the types are equal.
 */
function typeCmp(lhs: bril.Type, rhs: bril.Type): boolean {
  if (lhs === "any" || rhs === "any") {
    return true;
  } else if (lhs === "int" || lhs == "bool" || lhs == "float" || lhs == "char") {
    return lhs == rhs;
  } else {
    if (typeof rhs === "object" && Object.hasOwn(rhs, "ptr")) {
      return typeCmp(lhs.ptr, rhs.ptr);
    } else {
      return false;
    }
  }
}

function get(env: Env, ident: bril.Ident) {
  const val = env.get(ident);
  if (typeof val === "undefined") {
    throw error(`undefined variable ${ident}`);
  }
  return val;
}

function findFunc(func: bril.Ident, funcs: readonly bril.Function[]) {
  const matches = funcs.filter(function (f: bril.Function) {
    return f.name === func;
  });

  if (matches.length == 0) {
    throw error(`no function of name ${func} found`);
  } else if (matches.length > 1) {
    throw error(`multiple functions of name ${func} found`);
  }

  return matches[0];
}

function alloc(
  ptrType: bril.ParamType,
  amt: number,
  heap: Heap<Value>,
): Pointer {
  if (typeof ptrType != "object") {
    throw error(`unspecified pointer type ${ptrType}`);
  } else if (amt <= 0) {
    throw error(`must allocate a positive amount of memory: ${amt} <= 0`);
  } else {
    const loc = heap.alloc(amt);
    const dataType = ptrType.ptr;
    return {
      loc: loc,
      type: dataType,
    };
  }
}

/**
 * Ensure that the instruction has exactly `count` arguments,
 * throw an exception otherwise.
 */
function checkArgs(instr: bril.Operation, count: number) {
  const found = instr.args ? instr.args.length : 0;
  if (found != count) {
    throw error(`${instr.op} takes ${count} argument(s); got ${found}`);
  }
}

function getPtr(instr: bril.Operation, env: Env, index: number): Pointer {
  const val = getArgument(instr, env, index);
  if (typeof val !== "object" || val instanceof BigInt) {
    throw `${instr.op} argument ${index} must be a Pointer`;
  }
  return val;
}

function getArgument(
  instr: bril.Operation,
  env: Env,
  index: number,
  typ?: bril.Type,
) {
  const args = instr.args || [];
  if (args.length <= index) {
    throw error(
      `${instr.op} expected at least ${
        index + 1
      } arguments; got ${args.length}`,
    );
  }
  const val = get(env, args[index]);
  if (typ && !typeCheck(val, typ)) {
    throw error(`${instr.op} argument ${index} must be a ${typ}`);
  }
  return val;
}

function getInt(instr: bril.Operation, env: Env, index: number): bigint {
  return getArgument(instr, env, index, "int") as bigint;
}

function getBool(instr: bril.Operation, env: Env, index: number): boolean {
  return getArgument(instr, env, index, "bool") as boolean;
}

function getFloat(instr: bril.Operation, env: Env, index: number): number {
  return getArgument(instr, env, index, "float") as number;
}

function getChar(instr: bril.Operation, env: Env, index: number): string {
  return getArgument(instr, env, index, "char") as string;
}

function getLabel(instr: bril.Operation, index: number): bril.Ident {
  if (!instr.labels) {
    throw error(`missing labels; expected at least ${index + 1}`);
  }
  if (instr.labels.length <= index) {
    throw error(`expecting ${index + 1} labels; found ${instr.labels.length}`);
  }
  return instr.labels[index];
}

function getFunc(instr: bril.Operation, index: number): bril.Ident {
  if (!instr.funcs) {
    throw error(`missing functions; expected at least ${index + 1}`);
  }
  if (instr.funcs.length <= index) {
    throw error(
      `expecting ${index + 1} functions; found ${instr.funcs.length}`,
    );
  }
  return instr.funcs[index];
}

/**
 * The thing to do after interpreting an instruction: this is how `evalInstr`
 * communicates control-flow actions back to the top-level interpreter loop.
 */
type Action =
  | { "action": "next" } // Normal execution: just proceed to next instruction.
  | { "action": "jump"; "label": bril.Ident }
  | { "action": "end"; "ret": Value | null }
  | { "action": "speculate" }
  | { "action": "commit" }
  | { "action": "abort", "label": bril.Ident }
  | { "action": "break", "level": number } // Break out of n loops.
  | { "action": "continue", "level": number }; // Continue to next iteration of innermost loop.
const NEXT: Action = { "action": "next" };
const BREAK: Action = { "action": "break", "level": 0 };
const CONTINUE: Action = { "action": "continue", "level": 0 };

/**
 * The interpreter state that's threaded through recursive calls.
 */
type State = {
  env: Env;
  readonly heap: Heap<Value>;
  readonly funcs: readonly bril.Function[];

  // For profiling: a total count of the number of instructions executed.
  icount: bigint;

  // For SSA: the values of "shadow" variables set by set instructions.
  ssaEnv: Env;

  // For speculation: the state at the point where speculation began.
  specparent: State | null;
};

/**
 * Interpet a call instruction.
 */
function evalCall(instr: bril.Operation, state: State): Action {
  // Which function are we calling?
  const funcName = getFunc(instr, 0);
  const func = findFunc(funcName, state.funcs);
  if (func === null) {
    throw error(`undefined function ${funcName}`);
  }

  const newEnv: Env = new Map();

  // Check arity of arguments and definition.
  const params = func.args || [];
  const args = instr.args || [];
  if (params.length !== args.length) {
    throw error(
      `function expected ${params.length} arguments, got ${args.length}`,
    );
  }

  for (let i = 0; i < params.length; i++) {
    // Look up the variable in the current (calling) environment.
    const value = get(state.env, args[i]);

    // Check argument types
    if (!typeCheck(value, params[i].type)) {
      throw error(`function argument type mismatch`);
    }

    // Set the value of the arg in the new (function) environment.
    newEnv.set(params[i].name, value);
  }

  // Invoke the interpreter on the function.
  const newState: State = {
    env: newEnv,
    heap: state.heap,
    funcs: state.funcs,
    icount: state.icount,
    ssaEnv: new Map(),
    specparent: null, // Speculation not allowed.
  };
  const retVal = evalFunc(func, newState);
  state.icount = newState.icount;

  // Dynamically check the function's return value and type.
  if (!("dest" in instr)) { // `instr` is an `EffectOperation`.
    // Expected void function
    if (retVal !== null) {
      throw error(`unexpected value returned without destination`);
    }
    if (func.type !== undefined) {
      throw error(
        `non-void function (type: ${func.type}) doesn't return anything`,
      );
    }
  } else { // `instr` is a `ValueOperation`.
    // Expected non-void function
    if (instr.type === undefined) {
      throw error(`function call must include a type if it has a destination`);
    }
    if (instr.dest === undefined) {
      throw error(`function call must include a destination if it has a type`);
    }
    if (retVal === null) {
      throw error(
        `non-void function (type: ${func.type}) doesn't return anything`,
      );
    }
    if (!typeCheck(retVal, instr.type)) {
      throw error(
        `type of value returned by function does not match destination type`,
      );
    }
    if (func.type === undefined) {
      throw error(`function with void return type used in value call`);
    }
    if (!typeCmp(instr.type, func.type)) {
      throw error(
        `type of value returned by function does not match declaration`,
      );
    }
    state.env.set(instr.dest, retVal);
  }
  return NEXT;
}

/**
 * Interpret an instruction in a given environment, possibly updating the
 * environment. If the instruction branches to a new label, return that label;
 * otherwise, return "next" to indicate that we should proceed to the next
 * instruction or "end" to terminate the function.
 */
function evalInstr(instr: bril.Instruction, state: State): Action {
  state.icount += BigInt(1);

  // Check that we have the right number of arguments.
  if (instr.op !== "const") {
    const count = argCounts[instr.op];
    if (count === undefined) {
      throw error("unknown opcode " + instr.op);
    } else if (count !== null) {
      checkArgs(instr, count);
    }
  }

  // Function calls are not (currently) supported during speculation.
  // It would be cool to add, but aborting from inside a function call
  // would require explicit stack management.
  if (state.specparent && ["call", "ret"].includes(instr.op)) {
    throw error(`${instr.op} not allowed during speculation`);
  }

  switch (instr.op) {
    case "const": {
      // Interpret JSON numbers as either ints or floats.
      let value: Value;
      if (typeof instr.value === "number") {
        if (instr.type === "float") {
          value = instr.value;
        } else {
          value = BigInt(Math.floor(instr.value));
        }
      } else if (typeof instr.value === "string") {
        if ([...instr.value].length !== 1) {
          throw error(`char must have one character`);
        }
        value = instr.value;
      } else {
        value = instr.value;
      }

      state.env.set(instr.dest, value);
      return NEXT;
    }

    case "id": {
      const val = getArgument(instr, state.env, 0);
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "add": {
      let val = getInt(instr, state.env, 0) + getInt(instr, state.env, 1);
      val = BigInt.asIntN(64, val);
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "mul": {
      let val = getInt(instr, state.env, 0) * getInt(instr, state.env, 1);
      val = BigInt.asIntN(64, val);
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "sub": {
      let val = getInt(instr, state.env, 0) - getInt(instr, state.env, 1);
      val = BigInt.asIntN(64, val);
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "div": {
      const lhs = getInt(instr, state.env, 0);
      const rhs = getInt(instr, state.env, 1);
      if (rhs === BigInt(0)) {
        throw error(`division by zero`);
      }
      let val = lhs / rhs;
      val = BigInt.asIntN(64, val);
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "le": {
      const val = getInt(instr, state.env, 0) <= getInt(instr, state.env, 1);
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "lt": {
      const val = getInt(instr, state.env, 0) < getInt(instr, state.env, 1);
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "gt": {
      const val = getInt(instr, state.env, 0) > getInt(instr, state.env, 1);
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "ge": {
      const val = getInt(instr, state.env, 0) >= getInt(instr, state.env, 1);
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "eq": {
      const val = getInt(instr, state.env, 0) === getInt(instr, state.env, 1);
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "not": {
      const val = !getBool(instr, state.env, 0);
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "and": {
      const val = getBool(instr, state.env, 0) && getBool(instr, state.env, 1);
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "or": {
      const val = getBool(instr, state.env, 0) || getBool(instr, state.env, 1);
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "fadd": {
      const val = getFloat(instr, state.env, 0) + getFloat(instr, state.env, 1);
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "fsub": {
      const val = getFloat(instr, state.env, 0) - getFloat(instr, state.env, 1);
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "fmul": {
      const val = getFloat(instr, state.env, 0) * getFloat(instr, state.env, 1);
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "fdiv": {
      const val = getFloat(instr, state.env, 0) / getFloat(instr, state.env, 1);
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "fle": {
      const val =
        getFloat(instr, state.env, 0) <= getFloat(instr, state.env, 1);
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "flt": {
      const val = getFloat(instr, state.env, 0) < getFloat(instr, state.env, 1);
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "fgt": {
      const val = getFloat(instr, state.env, 0) > getFloat(instr, state.env, 1);
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "fge": {
      const val =
        getFloat(instr, state.env, 0) >= getFloat(instr, state.env, 1);
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "feq": {
      const val =
        getFloat(instr, state.env, 0) === getFloat(instr, state.env, 1);
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "print": {
      const args = instr.args || [];
      const values = args.map(function (i) {
        const val = get(state.env, i);
        if (val === UNDEF) {
          throw error(`print of undefined value`);
        }
        if (typeof val == "number") {
          if (Object.is(-0.0, val)) return "-" + val.toFixed(17);
          else if (val != 0.0 && Math.abs(Math.log10(Math.abs(val))) >= 10) {
            return val.toExponential(17);
          } else return val.toFixed(17);
        } else return val.toString();
      });
      console.log(...values);
      return NEXT;
    }

    case "jmp": {
      return { "action": "jump", "label": getLabel(instr, 0) };
    }

    case "br": {
      const cond = getBool(instr, state.env, 0);
      if (cond) {
        return { "action": "jump", "label": getLabel(instr, 0) };
      } else {
        return { "action": "jump", "label": getLabel(instr, 1) };
      }
    }

    case "ret": {
      const args = instr.args || [];
      if (args.length == 0) {
        return { "action": "end", "ret": null };
      } else if (args.length == 1) {
        const val = get(state.env, args[0]);
        return { "action": "end", "ret": val };
      } else {
        throw error(`ret takes 0 or 1 argument(s); got ${args.length}`);
      }
    }

    case "nop": {
      return NEXT;
    }

    case "call": {
      return evalCall(instr, state);
    }

    case "alloc": {
      const amt = getInt(instr, state.env, 0);
      const typ = instr.type;
      if (!(typeof typ === "object" && Object.hasOwn(typ, "ptr"))) {
        throw error(`cannot allocate non-pointer type ${instr.type}`);
      }
      const ptr = alloc(typ, Number(amt), state.heap);
      state.env.set(instr.dest, ptr);
      return NEXT;
    }

    case "free": {
      const val = getPtr(instr, state.env, 0);
      state.heap.free(val.loc);
      return NEXT;
    }

    case "store": {
      const target = getPtr(instr, state.env, 0);
      state.heap.write(
        target.loc,
        getArgument(instr, state.env, 1, target.type),
      );
      return NEXT;
    }

    case "load": {
      const ptr = getPtr(instr, state.env, 0);
      const val = state.heap.read(ptr.loc);
      if (val === undefined || val === null) {
        throw error(`Pointer ${instr.args![0]} points to uninitialized data`);
      } else {
        state.env.set(instr.dest, val);
      }
      return NEXT;
    }

    case "ptradd": {
      const ptr = getPtr(instr, state.env, 0);
      const val = getInt(instr, state.env, 1);
      state.env.set(instr.dest, {
        loc: ptr.loc.add(Number(val)),
        type: ptr.type,
      });
      return NEXT;
    }

    case "set": {
      const shadowVal = getArgument(instr, state.env, 1);
      const shadowDest = instr.args![0];
      state.ssaEnv.set(shadowDest, shadowVal);
      return NEXT;
    }

    case "get": {
      if (instr.args && instr.args.length) {
        throw error(`get should have zero arguments`);
      }

      // Get the "shadow" value for the destination variable.
      const val = state.ssaEnv.get(instr.dest);
      if (val === undefined) {
        throw error(`get without corresponding set for ${instr.dest}`);
      } else {
        state.env.set(instr.dest, val);
      }

      return NEXT;
    }

    case "undef": {
      state.env.set(instr.dest, UNDEF);
      return NEXT;
    }

    // Begin speculation.
    case "speculate": {
      return { "action": "speculate" };
    }

    // Abort speculation if the condition is false.
    case "guard": {
      if (getBool(instr, state.env, 0)) {
        return NEXT;
      } else {
        return { "action": "abort", "label": getLabel(instr, 0) };
      }
    }

    // Resolve speculation, making speculative state real.
    case "commit": {
      return { "action": "commit" };
    }

    case "ceq": {
      const val = getChar(instr, state.env, 0) === getChar(instr, state.env, 1);
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "clt": {
      const val = getChar(instr, state.env, 0) < getChar(instr, state.env, 1);
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "cle": {
      const val = getChar(instr, state.env, 0) <= getChar(instr, state.env, 1);
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "cgt": {
      const val = getChar(instr, state.env, 0) > getChar(instr, state.env, 1);
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "cge": {
      const val = getChar(instr, state.env, 0) >= getChar(instr, state.env, 1);
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "char2int": {
      const code = getChar(instr, state.env, 0).codePointAt(0);
      const val = BigInt.asIntN(64, BigInt(code as number));
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "int2char": {
      const i = getInt(instr, state.env, 0);
      if (i > 1114111 || i < 0 || (55295 < i && i < 57344)) {
        throw error(`value ${i} cannot be converted to char`);
      }
      const val = String.fromCodePoint(Number(i));
      state.env.set(instr.dest, val);
      return NEXT;
    }

    case "if": {
      const cond = getBool(instr, state.env, 0);
      if (cond) {
        const trueBody = instr.children?.[0];
        if (trueBody) {
          return evalBlock(trueBody, state);
        }
      } else {
        const falseBody = instr.children?.[1];
        if (falseBody) {
          return evalBlock(falseBody, state);
        }
      }
      return NEXT;
    }

    case "block": {
        const body = instr.children?.[0] || [];
        const result = evalBlock(body, state);
        if (result.action === "break") {
            if (result.level > 0) {
                return { action: "break", level: result.level - 1 };
            } else {
                return NEXT;
            }
        } else if (result.action === "continue") {
            if (result.level > 0) {
                return { action: "continue", level: result.level - 1};
            } else {
                return NEXT;
            }
        } else {
            return result;
        }
    }

    case "loop": {
      while (true) {
        const body = instr.children?.[0] || [];
        const result = evalBlock(body, state);
        if (result.action === "break") {
          if (result.level > 0) {
            return { action: "break", level: result.level - 1 };
          } else {
            return NEXT;
          }
        } else if (result.action === "continue") {
          if (result.level > 0) {
            return { action: "continue", level: result.level - 1 };
          } else {
            continue;
          }
        } else if (
          result.action === "end" ||
          result.action === "speculate" ||
          result.action === "commit" ||
          result.action === "abort"
        ) {
          return result;
        }
      }
    }

    case "break": {
      let level = instr.value;
      return { action: "break", level: level };
    }

    case "continue": {
      let level = instr.value;
      return { action: "continue", level: level };
    }
  }
  unreachable(instr);
  throw error(`unhandled opcode ${instr.op}`);
}

/**
 * Evaluate a block of instructions and return the resulting action.
 * This is a new function to handle blocks of instructions in the children array.
 */
function evalBlock(instrs: bril.Instruction[], state: State): Action {
  for (let i = 0; i < instrs.length; ++i) {
    const instr = instrs[i];
    if ("op" in instr) {
      const action = evalInstr(instr, state);

      switch (action.action) {
        case "next":
          break;
        case "break":
        case "continue":
        case "end":
        case "speculate":
        case "commit":
        case "abort":
          return action;
        case "jump":
            throw error(`label in structured control flow`);
        default:
          unreachable(action);
          throw error(`unhandled action ${action.action}`);
      }
    }
  }
  return NEXT;
}

function evalFunc(func: bril.Function, state: State): Value | null {
  for (let i = 0; i < func.instrs.length; ++i) {
    const line = func.instrs[i];
    if ("op" in line) {
      // Run an instruction.
      const action = evalInstr(line, state);

      // Take the prescribed action.
      switch (action.action) {
        case "end": {
          // Return from this function.
          return action.ret;
        }
        case "speculate": {
          // Begin speculation.
          state.specparent = { ...state };
          state.env = new Map(state.env);
          break;
        }
        case "commit": {
          // Resolve speculation.
          if (!state.specparent) {
            throw error(`commit in non-speculative state`);
          }
          state.specparent = null;
          break;
        }
        case "abort": {
          // Restore state.
          if (!state.specparent) {
            throw error(`abort in non-speculative state`);
          }
          // We do *not* restore `icount` from the saved state to ensure that we
          // count "aborted" instructions.
          Object.assign(state, {
            env: state.specparent.env,
            ssaEnv: state.specparent.ssaEnv,
            specparent: state.specparent.specparent,
          });
          break;
        }
        case "break": {
          // Break out of the innermost loop.
          if (state.specparent) {
            throw error(`break in speculative state`);
          }
          break;
        }
        case "continue": {
          // Continue to the next iteration of the innermost loop.
          if (state.specparent) {
            throw error(`continue in speculative state`);
          }
        }
        case "next":
        case "jump":
          break;
        default:
          unreachable(action);
          throw error(`unhandled action ${action.action}`);
      }
      // Move to a label.
      if ("label" in action) {
        // Search for the label and transfer control.
        for (i = 0; i < func.instrs.length; ++i) {
          const sLine = func.instrs[i];
          if ("label" in sLine && sLine.label === action.label) {
            --i; // Execute the label next.
            break;
          }
        }
        if (i === func.instrs.length) {
          throw error(`label ${action.label} not found`);
        }
      }
    }
  }

  // Reached the end of the function without hitting `ret`.
  if (state.specparent) {
    throw error(`implicit return in speculative state`);
  }
  return null;
}

function parseChar(s: string): string {
  const c = s;
  if ([...c].length === 1) {
    return c;
  } else {
    throw error(`char argument to main must have one character; got ${s}`);
  }
}

function parseBool(s: string): boolean {
  if (s === "true") {
    return true;
  } else if (s === "false") {
    return false;
  } else {
    throw error(`boolean argument to main must be 'true'/'false'; got ${s}`);
  }
}

function parseNumber(s: string): number {
  const f = parseFloat(s);
  // parseFloat and Number have subtly different behaviors for parsing strings
  // parseFloat ignores all random garbage after any valid number
  // Number accepts empty/whitespace only strings and rejects numbers with seperators
  // Use both and only accept the intersection of the results?
  const f2 = Number(s);
  if (!isNaN(f) && f === f2) {
    return f;
  } else {
    throw error(`float argument to main must not be 'NaN'; got ${s}`);
  }
}

function parseMainArguments(expected: bril.Argument[], args: string[]): Env {
  const newEnv: Env = new Map();

  if (args.length !== expected.length) {
    throw error(
      `mismatched main argument arity: expected ${expected.length}; got ${args.length}`,
    );
  }

  for (let i = 0; i < args.length; i++) {
    const type = expected[i].type;
    switch (type) {
      case "int": {
        // https://dev.to/darkmavis1980/you-should-stop-using-parseint-nbf
        const n: bigint = BigInt(Number(args[i]));
        newEnv.set(expected[i].name, n as Value);
        break;
      }
      case "float": {
        const f: number = parseNumber(args[i]);
        newEnv.set(expected[i].name, f as Value);
        break;
      }
      case "bool": {
        const b: boolean = parseBool(args[i]);
        newEnv.set(expected[i].name, b as Value);
        break;
      }
      case "char": {
        const c: string = parseChar(args[i]);
        newEnv.set(expected[i].name, c as Value);
        break;
      }
    }
  }
  return newEnv;
}

function evalProg(prog: bril.Program) {
  const heap = new Heap<Value>();
  const main = findFunc("main", prog.functions);
  if (main === null) {
    console.warn(`no main function defined, doing nothing`);
    return;
  }

  // Silly argument parsing to find the `-p` flag.
  const args: string[] = Array.from(Deno.args);
  let profiling = false;
  const pidx = args.indexOf("-p");
  if (pidx > -1) {
    profiling = true;
    args.splice(pidx, 1);
  }

  // Remaining arguments are for the main function.k
  const expected = main.args || [];
  const newEnv = parseMainArguments(expected, args);

  const state: State = {
    funcs: prog.functions,
    heap,
    env: newEnv,
    icount: BigInt(0),
    ssaEnv: new Map(),
    specparent: null,
  };
  evalFunc(main, state);

  if (!heap.isEmpty()) {
    throw error(
      `Some memory locations have not been freed by end of execution.`,
    );
  }

  if (profiling) {
    console.error(`total_dyn_inst: ${state.icount}`);
  }
}

async function main() {
  try {
    const prog = JSON.parse(await readStdin()) as bril.Program;
    evalProg(prog);
  } catch (e) {
    if (e instanceof BriliError) {
      console.error(`error: ${e.message}`);
      Deno.exit(2);
    } else {
      throw e;
    }
  }
}

main();
