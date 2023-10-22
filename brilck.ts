import * as bril from './bril-ts/bril.ts';
import {Signature, PolySignature, FuncType, OP_SIGS, TVar, BaseSignature, PolyType} from './bril-ts/types.ts';
import {readStdin, unreachable} from './bril-ts/util.ts';

/**
 * The JavaScript types of Bril constant values.
 */
const CONST_TYPES: {[key: string]: string} = {
  'int': 'number',
  'float': 'number',
  'bool': 'boolean',
  'char': 'string'
};

type VarEnv = Map<bril.Ident, bril.Type>;
type FuncEnv = Map<bril.Ident, FuncType>;
type TypeEnv = Map<string, bril.Type>;

/**
 * A typing environment that we can use to check instructions within
 * a single function.
 */
interface Env {
  /**
   * The types of all variables defined in the function.
   */
  vars: VarEnv;

  /**
   * The names of all the labels in the function.
   */
  labels: Set<bril.Ident>;

  /**
   * The defined functions in the program.
   */
  funcs: FuncEnv;

  /**
   * The return type of the current function.
   */
  ret: bril.Type | undefined;
}

/**
 * An optional filename for error messages.
 */
let CHECK_FILE: string | undefined;

/**
 * The total number of errors we encounter.
 */
let ERRORS: number = 0;

/**
 * Print an error message, possibly with a source position.
 */
function err(msg: string, pos: bril.Position | undefined) {
  ERRORS++;
  if (pos) {
    msg = `${pos.row}:${pos.col}: ${msg}`;
  }
  if (CHECK_FILE) {
    msg = `${CHECK_FILE}:${msg}`;
  }
  console.error(msg);
}

/**
 * Set the type of variable `id` to `type` in `env`, checking for conflicts
 * with the old type for the variable.
 */
function addType(env: VarEnv, id: bril.Ident, type: bril.Type, pos: bril.Position | undefined) {
  let oldType = env.get(id);
  if (oldType) {
    if (!typeEq(oldType, type)) {
      err(
        `new type ${type} for ${id} conflicts with old type ${oldType}`,
        pos
      );
    }
  } else {
    env.set(id, type);
  }
}

/**
 * Look up type variables in TypeEnv, leaving non-variable types and undefined
 * type variables unchanged.
 */
function typeLookup(type: PolyType, tenv: TypeEnv | undefined): PolyType {
  if (!tenv) {
    return type;
  }

  // Do we have a type variable to look up?
  if (typeof type === 'object' && 'tv' in type) {
    let res = tenv.get(type.tv);
    if (res) {
      return res;
    } else {
      return type;
    }
  }

  // Do we need to recursively look up inside this type?
  if (typeof type === 'object' && 'ptr' in type) {
    return {ptr: typeLookup(type.ptr, tenv)};
  }

  return type;
}

/**
 * Check for type equality.
 *
 * If a type environemnt is supplied, attempt to unify any unset type
 * variables occuring in `b` to make the types match.
 */
function typeEq(a: bril.Type, b: PolyType, tenv?: TypeEnv): boolean {
  // Shall we bind a type variable in b?
  b = typeLookup(b, tenv);
  if (typeof b === "object" && 'tv' in b) {
    if (!tenv) {
      throw `got type variable ${b.tv} but no type environment`;
    }
    tenv.set(b.tv, a);
    return true;
  }

  // Normal type comparison.
  if (typeof a === "string" && typeof b === "string") {
    return a == b;
  } else if (typeof a === "object" && typeof b === "object") {
    return typeEq(a.ptr, b.ptr, tenv);
  } else {
    return false;
  }
}

/**
 * Format a type as a human-readable string.
 */
function typeFmt(t: PolyType): string {
  if (typeof t === "string") {
    return t;
  } else if (typeof t === "object") {
    if ('tv' in t) {
      return t.tv;
    } else {
      return `ptr<${typeFmt(t.ptr)}>`;
    }
  }
  unreachable(t);
}

/**
 * Check an instruction's arguments and labels against a type signature.
 *
 * `sig` may be either a concrete signature or a polymorphic one, in which case
 * we try unify the quantified type. `name` optionally gives a name for the
 * operation to use in error messages; otherwise, we use `instr`'s opcode.
 */
function checkSig(env: Env, instr: bril.Operation, psig: Signature | PolySignature, name?: string) {
  name = name ?? instr.op;

  // Are we handling a polymorphic signature?
  let sig: BaseSignature<PolyType>;
  let tenv: TypeEnv = new Map();
  if ('tvar' in psig) {
    sig = psig.sig;
  } else {
    sig = psig;
  }

  // Check destination type.
  if ('type' in instr) {
    if (sig.dest) {
      if (!typeEq(instr.type, sig.dest, tenv)) {
        err(
          `result type of ${name} should be ${typeFmt(typeLookup(sig.dest, tenv))}, ` +
          `but found ${typeFmt(instr.type)}`,
          instr.pos
        );
      }
    } else {
      err(`${name} should have no result type`, instr.pos);
    }
  } else {
    if (sig.dest) {
      err(
        `missing result type ${typeFmt(typeLookup(sig.dest, tenv))} for ${name}`,
        instr.pos
      );
    }
  }

  // Check arguments.
  let args = instr.args ?? [];
  if (args.length !== sig.args.length) {
    err(
      `${name} expects ${sig.args.length} args, not ${args.length}`,
      instr.pos
    );
  } else {
    for (let i = 0; i < args.length; ++i) {
      let argType = env.vars.get(args[i]);
      if (!argType) {
        err(`${args[i]} (arg ${i}) undefined`, instr.pos);
        continue;
      }
      if (!typeEq(argType, sig.args[i], tenv)) {
        err(
          `${args[i]} has type ${typeFmt(argType)}, but arg ${i} for ${name} ` +
          `should have type ${typeFmt(typeLookup(sig.args[i], tenv))}`,
          instr.pos
        );
      }
    }
  }

  // Check labels.
  let labs = instr.labels ?? [];
  let labCount = sig.labels ?? 0;
  if (labs.length !== labCount) {
    err(`${instr.op} needs ${labCount} labels; found ${labs.length}`, instr.pos);
  } else {
    for (let lab of labs) {
      if (!env.labels.has(lab)) {
        err(`label .${lab} undefined`, instr.pos);
      }
    }
  }
}

type CheckFunc = (env: Env, instr: bril.Operation) => void;

/**
 * Special-case logic for checking some special functions.
 */
const INSTR_CHECKS: {[key: string]: CheckFunc} = {
  print: (env, instr) => {
    if ('type' in instr) {
      err(`print should have no result type`, instr.pos);
    }
  },

  call: (env, instr) => {
    let funcs = instr.funcs ?? [];
    if (funcs.length !== 1) {
      err(`call should have one function, not ${funcs.length}`, instr.pos);
      return;
    }

    let funcType = env.funcs.get(funcs[0]);
    if (!funcType) {
      err(`function @${funcs[0]} undefined`, instr.pos);
      return;
    }

    checkSig(env, instr, {
      args: funcType.args,
      dest: funcType.ret,
    }, `@${funcs[0]}`);
    return;
  },

  ret: (env, instr) => {
    let args = instr.args ?? [];
    if (env.ret) {
      if (args.length === 0) {
        err(`missing return value in function with return type`, instr.pos);
      } else if (args.length !== 1) {
        err(`cannot return multiple values`, instr.pos);
      } else {
        checkSig(env, instr, {args: [env.ret]});
      }
    } else {
      if (args.length !== 0) {
        err(`returning value in function without a return type`, instr.pos);
      }
    }
    return;
  },

  phi: (env, instr) => {
    let args = instr.args ?? [];
    if (!('type' in instr)) {
      err(`phi needs a result type`, instr.pos);
      return;
    }

    // Construct a signature with uniform argument types.
    let argTypes: bril.Type[] = [];
    for (let i = 0; i < args.length; ++i) {
      argTypes.push(instr.type);
    }
    checkSig(env, instr, {args: argTypes, dest: instr.type, labels: args.length});
  },
};

function checkOp(env: Env, instr: bril.Operation) {
  let args = instr.args ?? [];

  // Check for special cases.
  let check_func = INSTR_CHECKS[instr.op];
  if (check_func) {
    check_func(env, instr);
    return;
  }

  // General case: use the operation's signature.
  let sig = OP_SIGS[instr.op];
  if (!sig) {
    err(`unknown opcode ${instr.op}`, instr.pos);
    return;
  }
  checkSig(env, instr, sig);
}

function checkConst(instr: bril.Constant) {
  if (!(instr as any)) {
    err(`const missing type`, instr!.pos);
    return;
  }

  if (typeof instr.type !== 'string') {
    err(`const of non-primitive type ${typeFmt(instr.type)}`, instr.pos);
    return;
  }

  let valType = CONST_TYPES[instr.type];
  if (!valType) {
    err(`unknown const type ${typeFmt(instr.type)}`, instr.pos);
    return;
  }

  if (typeof instr.value !== valType) {
    err(
      `const value ${instr.value} does not match type ${typeFmt(instr.type)}`,
      instr.pos
    );
  }
}

function checkFunc(funcs: FuncEnv, func: bril.Function) {
  let vars: VarEnv = new Map();
  let labels = new Set<bril.Ident>();

  // Initilize the type environment with the arguments.
  if (func.args) {
    for (let arg of func.args) {
      addType(vars, arg.name, arg.type, func.pos);
    }
  }

  // Gather up all the types of the local variables and all the label names.
  if (func.instrs){
    for (let instr of func.instrs) {
      if ('dest' in instr) {
        addType(vars, instr.dest, instr.type, instr.pos);
      } else if ('label' in instr) {
        if (labels.has(instr.label)) {
          err(`multiply defined label .${instr.label}`, instr.pos);
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
          checkOp({vars, labels, funcs, ret: func.type}, instr);
        }
      }
    }
  }
}

function checkProg(prog: bril.Program) {
  // Gather up function types.
  let funcEnv: FuncEnv = new Map();
  for (let func of prog.functions) {
    funcEnv.set(func.name, {
      ret: func.type,
      args: func.args?.map(a => a.type) ?? [],
    });
  }

  // Check each function.
  for (let func of prog.functions) {
    checkFunc(funcEnv, func);

    // The @main function must not return anything.
    if (func.name === 'main') {
      if (func.type) {
        err(`@main must have no return type; found ${typeFmt(func.type)}`,
            func.pos);
      }
    }
  }
}

async function main() {
  if (Deno.args[0]) {
    CHECK_FILE = Deno.args[0];
  }
  let prog = JSON.parse(await readStdin()) as bril.Program;
  checkProg(prog);
  if (ERRORS) {
    Deno.exit(1);
  }
}

main();
