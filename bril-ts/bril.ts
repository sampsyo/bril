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
export type Type = "int" | "bool" | "float" | "double";

/**
 * An instruction that does not produce any result.
 */
export interface EffectOperation {
  op: "br" | "jmp" | "print" | "ret" | "obv";
  args: Ident[];
}

/**
 * An operation that produces a value and places its result in the
 * destination variable.
 */
interface ValueOperationCore {
  args: Ident[];
  dest: Ident;
  type: Type;
}
 
export interface DetValueOperation extends ValueOperationCore {
  op: "add" | "mul" | "sub" | "div" |
      "id" | "nop" |
      "eq" | "lt" | "gt" | "ge" | "le" | "not" | "and" | "or" |
      "fadd" | "fmul" | "fsub" | "fdiv" |
      "feq" | "flt" | "fgt" | "fge" | "fle";
}

/**
 * An instruction for random 
 TODO: figure out what counts as RandOperation when we add Normal, or
 int-supported distributions?
 */
export interface RandValueOperation extends ValueOperationCore {
  op: "flip";
}

export type ValueOperation = DetValueOperation | RandValueOperation;

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
export type Operation = EffectOperation | ValueOperation ;

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

/**
 * A function consists of a sequence of instructions.
 */
export interface Function {
  name: Ident;
  instrs: (Instruction | Label)[];
}

/**
 * A program consists of a set of functions, one of which must be named
 * "main".
 */
export interface Program {
  functions: Function[];
}
