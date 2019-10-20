open Core
open Cfg

let assigns_to var instr =
  match instr with
  | ConstInstr {dest; _}
  | ValueInstr {dest; _} ->
     dest = var
  | _ -> false

let instr_transparent expression instr =
  let doesnt_assign_to var =
    not (assigns_to var instr)
  in
  List.for_all ~f:doesnt_assign_to @@ Bril.args expression

let instrs_transparent expression instrs =
  List.for_all ~f:(instr_transparent expression) instrs

let transparent expression block =
  instrs_transparent expression block.body

let instr_computes expression instr =
  match instr with
  | Cfg.ValueInstr { op; _ } ->
     op = expression
  | _ -> false

let computes expression block =
  let rec computes' instrs =
    match instrs with
    | [] -> false
    | instr :: instrs ->
       if instr_computes expression instr
       then instrs_transparent expression instrs
       else computes' instrs
  in
  computes' block.body

let anticipates expression block =
  computes expression {block with body = List.rev block.body}

let expression = Bril.Add (Ident.var_of_string "x",
                           Ident.var_of_string "y")

module Availability =
  Graph.Fixpoint.Make(CFG)
    (struct 
      type vertex = CFG.E.vertex
      type edge = CFG.E.t
      type g = CFG.t
      type data = bool
      let direction = Graph.Fixpoint.Forward
      let equal = (=)
      let join = (&&)
      let analyze (src, _) src_avail_in =
        transparent expression src
        && src_avail_in
        || computes expression src
    end)

module Anticipatability =
  Graph.Fixpoint.Make(CFG)
    (struct 
      type vertex = CFG.E.vertex
      type edge = CFG.E.t
      type g = CFG.t
      type data = bool
      let direction = Graph.Fixpoint.Backward
      let equal = (=)
      let join = (&&)
      let analyze (_, dst) dst_ant_out =
        transparent expression dst
        && dst_ant_out
        || anticipates expression dst
    end)
    
    
  
