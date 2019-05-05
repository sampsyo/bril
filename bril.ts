export type Ident = string;

export const enum Operation {
  add = "add",
  id = "id",
}

export interface Instruction {
  op: Operation;
  args: Ident[];
  dest: Ident;
}

export interface Function {
  name: Ident;
  instrs: Instruction[];
}

export interface Program {
  functions: Function[];
}
