export type Operation = "add" | "id";
export type Ident = string;

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
