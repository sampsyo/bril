/**
 * The definition of the Bril language.
 */

/**
 * A variable name.
 */
export type Ident = string;

/**
 * Primitive types.
 */
export type PrimType = "int" | "bool" | "float";

/**
 * Parameterized types.
 */
export type ParamType = {"ptr": Type, "product": Type[], "sum": Type[]};

/**
 * Value types.
 */
export type Type = PrimType | ParamType;

/**
 * Common fields in any operation.
 */
interface Op {
  args?: Ident[];
  funcs?: Ident[];
  labels?: Ident[];
}

/**
 * An instruction that does not produce any result.
 */
export interface EffectOperation extends Op {
  op: "br" | "jmp" | "print" | "ret" | "call" |
    "store" | "free" |
    "speculate" | "guard" | "commit";
}

/**
 * An operation that produces a value and places its result in the
 * destination variable.
 */
export interface ValueOperation extends Op {
  op: "add" | "mul" | "sub" | "div" |
      "id" | "nop" |
      "eq" | "lt" | "gt" | "ge" | "le" | "not" | "and" | "or" |
      "call" |
      "load" | "ptradd" | "alloc" |
      "fadd" | "fmul" | "fsub" | "fdiv" |
      "feq" | "flt" | "fle" | "fgt" | "fge" |
      "phi" |
      "pack" | "unpack" | "construct" | "destruct";
  dest: Ident;
  type: Type;
}

/**
 * The type of Bril values that may appear in constants.
 */
export type Value = number | boolean;

/**
 * An instruction that places a literal value into a variable.
 */
export interface Constant {
  op: "const";
  value: Value;
  dest: Ident;
  type: Type;
}


/**
 * Operations take arguments, which come from previously-assigned identifiers.
 */
export type Operation = EffectOperation | ValueOperation;

/**
 * Instructions can be operations (which have arguments) or constants (which
 * don't). Both produce a value in a destination variable.
 */
export type Instruction = Operation | Constant;

/**
 * Both constants and value operations produce results.
 */
export type ValueInstruction = Constant | ValueOperation;

/**
 * The valid opcodes for value-producing instructions.
 */
export type ValueOpCode = ValueOperation["op"];

/**
 * The valid opcodes for effecting operations.
 */
export type EffectOpCode = EffectOperation["op"];

/**
 * All valid operation opcodes.
 */
export type OpCode = ValueOpCode | EffectOpCode;

/**
 * Jump labels just mark a position with a name.
 */
export interface Label {
  label: Ident;
}

/*
 * An argument has a name and a type.
 */
export interface Argument {
  name: Ident;
  type: Type;
}

/**
 * A function consists of a sequence of instructions.
 */
export interface Function {
  name: Ident;
  args?: Argument[];
  instrs: (Instruction | Label)[];
  type?: Type;
}

/**
 * A program consists of a set of functions, one of which must be named
 * "main".
 */
export interface Program {
  functions: Function[];
}
