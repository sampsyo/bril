/**
 * The definition of the Bril language.
 */

/**
 * A variable name.
 */
export type Ident = string;

/**
 * Value types.
 */
export type Type = "int" | "bool";

/**
 * An instruction that does not produce any result.
 */
export interface EffectOperation {
  op: "br" | "jmp" | "print" | "ret";
  args: Ident[];
}

/**
 * An operation that produces a value and places its result in the
 * destination variable.
 */
export interface ValueOperation {
  op: "add" | "mul" | "sub" | "div" |
      "id" | "nop" |
      "eq" | "lt" | "gt" | "ge" | "le" | "not" | "and" | "or";
  args: Ident[];
  dest: Ident;
  type: Type;
}

/**
 * An operation that calls another Bril function which has a return type and 
 * stores the result in the destination variable.
 */
export interface ValueCallOperation {
  op: "call";
  name: Ident;
  args: Ident[];
  dest: Ident;
  type: Type; 
}

/**
 * An operation that calls another Bril function with a void return type.
 */
export interface EffectCallOperation {
  op: "call";
  name: Ident;
  args: Ident[];
}


/**
 * An operation that calls another Bril function.
 */
export type CallOperation = ValueCallOperation | EffectCallOperation;

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
export type Operation = EffectOperation | ValueOperation | CallOperation;

/**
 * Instructions can be operations (which have arguments) or constants (which
 * don't). Both produce a value in a destination variable.
 */
export type Instruction = Operation | Constant;

/**
 * Both constants and value operations produce results.
 */
export type ValueInstruction = Constant | ValueOperation | ValueCallOperation;

/**
 * The valid opcodes for value-producing instructions.
 */
export type ValueOpCode = ValueOperation["op"];

/**
 * The valid opcodes for effecting operations.
 */
export type EffectOpCode = EffectOperation["op"];

/**
 * The valid opcode for call operations.
 */
export type CallOpCode = CallOperation["op"];

/**
 * All valid operation opcodes.
 */
export type OpCode = ValueOpCode | EffectOpCode | CallOpCode;

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
  args: Argument[];
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
