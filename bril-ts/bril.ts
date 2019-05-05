/**
 * The definition of the Bril language.
 */

/**
 * A variable name.
 */
export type Ident = string;

/**
 * The choice of operation for an `Operation` instruction.
 */
export const enum OpCode {
  add = "add",
  id = "id",
  print = "print",
}

/**
 * An instruction that manipulates arguments, which come from
 * previously-assigned identifiers, and places a result in the destination
 * variable.
 */
export interface Operation {
  op: OpCode;
  args: Ident[];
  dest: Ident;
}

/**
 * The type of Bril values that may appear in constants.
 */
export type Value = number;

/**
 * An instruction that places a literal value into a variable.
 */
export interface Const {
  op: "const";
  value: Value;
  dest: Ident;
}

/**
 * Instructions can be operations (which have arguments) or constants (which
 * don't). Both produce a value in a destination variable.
 */
export type Instruction = Operation | Const;

/**
 * A function consists of a sequence of instructions.
 */
export interface Function {
  name: Ident;
  instrs: Instruction[];
}

/**
 * A program consists of a set of functions, one of which must be named
 * "main".
 */
export interface Program {
  functions: Function[];
}
