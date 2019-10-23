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

export interface TraceEffectOperation {
  op: "trace",
  failLabel: Ident;
  effect: EffectOperation,
  args: Ident[],
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
 * A group is a list of instructions and represents a VLIW (Very Long Instruction Word):
 * multiple instructions that can run at the same time. The first list represents
 * set of conditions to be executed. The execution jumps to `failLabel` if
 * the conditional is false.
 */
export type Group = {
  conds: Ident[];
  instrs: (ValueOperation | Constant)[];
  failLabel: Ident;
}

/**
 * Micro-instructions can be operations (which have arguments) or constants (which
 * don't). Both produce a value in a destination variable.
 */
export type MicroInstruction = Operation | Constant | TraceEffectOperation;

/**
 * Represents a thing that can execute at the same time. It is either a single MicroInstruction
 * or a group which is an Array of MicroInstructions.
 */
export type Instruction = Group | MicroInstruction;


export function logInstr(instr: Instruction | Label) {
  if ("conds" in instr) {
    console.log(instr);
  } else if ("label" in instr) {
    console.log(instr.label)
  } else {
    switch (instr.op) {
      case "br":
        console.log("  br", instr.args[0], instr.args[1], instr.args[2]);
        break;
      case "jmp":
        console.log("  jmp", instr.args[0]);
        break;
      case "ret":
        console.log("  ret");
        break;
      case "const":
        console.log(" ", instr.dest, "=", "const", instr.value);
        break;
      case "trace":
        process.stdout.write(`  trace ${instr.failLabel} [ ${instr.args} ] <-`);
        logInstr(instr.effect);
        break;
      default:
        if ("dest" in instr) {
          console.log(" ", instr.dest, "=", instr.op, instr.args);
        } else {
          console.log(instr);
        }
    }
  }
}

export function logInstrs(instrs: (Label | Instruction)[]) {
  instrs.forEach((v) => logInstr(v));
}

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
