open Graph
open Core

module Ident : sig
  type var
  type lbl
  type fn
  val cmp_var : var -> var -> int
  val cmp_lbl : lbl -> lbl -> int
  val cmp_fn : fn -> fn -> int
end = struct
  type var = string
  type lbl = string
  type fn = string
  let cmp_var = String.compare
  let cmp_lbl = String.compare
  let cmp_fn = String.compare
end

module Bril = struct

  type typ =
    | Int
    | Bool

  type value =
    | Int of int
    | Bool of bool

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

  type value_op =
    { op: value_expr;
      dest: Ident.var;
      typ: typ }

  type term_op =
    | Br of {cond: Ident.var; true_lbl: Ident.lbl; false_lbl: Ident.lbl}
    | Jmp of Ident.lbl
    | Ret

  type effect_op =
    | TermOp of term_op
    | Print of Ident.var list

  type const_op =
    { dest: Ident.var;
      value: value;
      typ: typ }

  type instruction =
    | ValueInstr of value_op
    | EffectInstr of effect_op
    | ConstInstr of const_op

  type directive =
    | Instruction of instruction
    | Label of Ident.lbl

  type fn =
    { name: Ident.fn;
      body: directive list }
              
  type program = fn list
end
                  
module CFG = struct

  type instruction = 
    | ValueInstr of Bril.value_op
    | Print of Ident.var list
    | ConstInstr of Bril.const_op

  type basic_block =
    { label: Ident.lbl;
      body: instruction list;
      term: Bril.term_op }

  module BasicBlock = struct
    type t = basic_block
    let compare x y =
      Ident.cmp_lbl x.label y.label
    let hash x =
      Hashtbl.hash x.label
    let equal x y =
      x.label = y.label
  end

  include Persistent.Digraph.Concrete(BasicBlock)
end
           
let fu _ =
  failwith "unimplemented"

let parse_bril (file: In_channel.t): Bril.program =
  fu file

let build_cfg (prog: Bril.program) : CFG.t =
  fu prog

let analyze (cfg: CFG.t) =
  fu cfg

let go file =
  file |> parse_bril |> build_cfg |> analyze

let command =
  let spec =
    let open Command.Spec in
    empty +> anon (maybe ("brilfile" %:string))
  in
  Command.basic_spec
    ~summary:"lcm: Lazy code motion for the Bril intermediate language"
    spec
    (function
     | Some path -> go @@ In_channel.create path
     | None -> go In_channel.stdin)

let () =
  Command.run ~version:"0.1.1" command
