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
   * Build an operation instruction that produces a result. If the name is
   * omitted, a fresh variable is chosen automatically.
   */
  buildValue(op: bril.ValueOpCode, args: string[],
             type: bril.Type, dest?: string) {
    dest = dest || this.freshVar();
    let instr: bril.ValueOperation = { op, args, dest, type };
    this.insert(instr);
    return instr;
  }

  /**
   * Build a non-value-producing (side-effecting) operation instruction.
   */
  buildEffect(op: bril.EffectOpCode, args: string[]) {
    let instr: bril.EffectOperation = { op, args };
    this.insert(instr);
    return instr;
  }

  /**
   * Build a constant instruction. As above, the destination name is optional.
   */
  buildConst(value: bril.Value, type: bril.Type, dest?: string) {
    dest = dest || this.freshVar();
    let instr: bril.Constant = { op: "const", value, dest, type };
    this.insert(instr);
    return instr;
  }

  /**
   * Build a constant integer value.
   */
  buildInt(value: number, dest?: string) {
    return this.buildConst(value, "int", dest);
  }

  /**
   * Build a constant boolean value.
   */
  buildBool(value: boolean, dest?: string) {
    return this.buildConst(value, "bool", dest);
  }

  /**
   * Add a label to the function at the current position.
   */
  buildLabel(name: string) {
    let label = {label: name};
    this.insert(label);
  }

  /**
   * Insert an instruction at the end of the current function.
   */
  private insert(instr: bril.Instruction | bril.Label) {
    if (!this.curFunction) {
      throw "cannot build instruction/label without a function";
    }
    this.curFunction.instrs.push(instr);
  }

  /**
   * Generate an unused variable name.
   */
  freshVar() {
    let out = 'v' + this.nextFresh.toString();
    this.nextFresh += 1;
    return out;
  }

  /**
   * Generate an unused suffix.
   */
  freshSuffix() {
    let out = '.' + this.nextFresh.toString();
    this.nextFresh += 1;
    return out;
  }
}
