open Core
   
type typ =
  | Int
  | Bool
[@@deriving sexp]

type value =
  | Int of int
  | Bool of bool
[@@deriving sexp]

type value_expr =
  | Add of Ident.var * Ident.var
  | Mul of Ident.var * Ident.var
  | Sub of Ident.var * Ident.var
  | Div of Ident.var * Ident.var
  | Eq of Ident.var * Ident.var
  | Lt of Ident.var * Ident.var
  | Gt of Ident.var * Ident.var
  | Le of Ident.var * Ident.var
  | Ge of Ident.var * Ident.var
  | Not of Ident.var
  | And of Ident.var * Ident.var
  | Or of Ident.var * Ident.var
  | Id of Ident.var
  | Nop
[@@deriving sexp]

let args = function
  | Add (x, y) -> [x; y]
  | Mul (x, y) -> [x; y]
  | Sub (x, y) -> [x; y]
  | Div (x, y) -> [x; y]
  | Eq (x, y) -> [x; y]
  | Lt (x, y) -> [x; y]
  | Gt (x, y) -> [x; y]
  | Le (x, y) -> [x; y]
  | Ge (x, y) -> [x; y]
  | Not x -> [x]
  | And (x, y) -> [x; y]
  | Or (x, y) -> [x; y]
  | Id x -> [x]
  | Nop -> []

type value_op =
  { op: value_expr;
    dest: Ident.var;
    typ: typ }
[@@deriving sexp]

type term_op =
  | Br of {cond: Ident.var; true_lbl: Ident.lbl; false_lbl: Ident.lbl}
  | Jmp of Ident.lbl
  | Ret
[@@deriving sexp]

type effect_op =
  | TermOp of term_op
  | Print of Ident.var list
[@@deriving sexp]

type const_op =
  { dest: Ident.var;
    value: value;
    typ: typ }
[@@deriving sexp]

type instruction =
  | ValueInstr of value_op
  | EffectInstr of effect_op
  | ConstInstr of const_op
  | Nop
[@@deriving sexp]

type directive =
  | Instruction of instruction
  | Label of Ident.lbl
[@@deriving sexp]

type fn =
  { name: Ident.fn;
    body: directive list }
[@@deriving sexp]
  
type program = fn list [@@deriving sexp]
