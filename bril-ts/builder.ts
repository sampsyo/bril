import * as bril from './bril';

/**
 * A utility for building up Bril programs.
 */
export class Builder {
  /**
   * The program we have built so far.
   */
  public program: bril.Program = { functions: [] };

  private curFunction: bril.Function | null = null;
  private nextFresh: number = 0;

  /**
   * Create a new, empty function into which further code will be generated.
   */
  buildFunction(name: string) {
    let func: bril.Function = { name, instrs: [] };
    this.program.functions.push(func);
    this.curFunction = func;
    this.nextFresh = 0;
    return func;
  }

  /**
   * Build an operation instruction. If the name is omitted, a fresh variable
   * is chosen automatically.
   */
  buildOp(op: bril.OpCode, args: string[], dest?: string) {
    dest = dest || this.fresh();
    let instr: bril.Operation = { op, args, dest };
    this.insertInstr(instr);
    return instr;
  }

  /**
   * Build a constant instruction. As above, the destination name is optional.
   */
  buildConst(value: bril.Value, dest?: string) {
    dest = dest || this.fresh();
    let instr: bril.Const = { op: "const", value, dest };
    this.insertInstr(instr);
    return instr;
  }

  /**
   * Insert an instruction at the end of the current function.
   */
  private insertInstr(instr: bril.Instruction) {
    if (!this.curFunction) {
      throw "cannot build instruction without a function";
    }
    this.curFunction.instrs.push(instr);
  }

  /**
   * Generate an unused variable name.
   */
  private fresh() {
    let out = '%' + this.nextFresh.toString();
    this.nextFresh += 1;
    return out;
  }
}

