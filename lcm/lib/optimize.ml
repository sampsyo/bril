open Core
open Cfg

let index_of x l =
  let rec go l n =
    match l with
    | [] -> -1
    | y :: l ->
       if y = x
       then n
       else go l (n + 1)
  in
  go l 0

let rec instrs_rewrite ~f instrs =
  match instrs with
  | instr :: instrs ->
     f instr @ instrs_rewrite ~f instrs
  | [] -> []

let block_instr_rewrite ~f (block, meta) =
  let body' = instrs_rewrite ~f block.body in
  { block with body = body' }, meta

let instr_rewrite ~f graph =
  CFG.map_vertex (block_instr_rewrite ~f) graph

let expr_loc exprs expr = 
  "_lcm_tmp" ^ string_of_int (index_of expr exprs)
  |> Ident.var_of_string 

let unify_expression_locations exprs graph =
  let fix_computation = function
    | ValueInstr { op; dest; typ } ->
       if Bril.is_computation op
       then let loc = expr_loc exprs op in
            [ValueInstr { op; dest = loc; typ };
             ValueInstr { op = Id loc; dest = dest; typ }]
       else [ValueInstr { op; dest; typ }]
    | i -> [i]
  in
  instr_rewrite ~f:fix_computation graph

let interp_bitv exprs bitv =
  Bitv.foldi_left
    (fun accr idx bit ->
      if bit
      then List.nth_exn exprs idx :: accr
      else accr)
    [] bitv

let exprs_equal x y =
  Bril.compare_value_expr x y = 0

let delete_computations exprs_to_delete block =
  let found = ref false in (* hack *)
  block_instr_rewrite block
    ~f:(fun i ->
      match i with
      | ValueInstr { op; _ } ->
         if not !found && List.mem exprs_to_delete op ~equal:exprs_equal
         then begin found := true; [] end
         else [i]
      | _ -> [i])

let delete_computations exprs graph =
  CFG.map_vertex
    (fun block ->
      let exprs_to_delete =
        interp_bitv exprs @@ Attrs.get (snd block) "delete"
      in
      Printf.printf "%s\n" (Ident.string_of_lbl (fst block).lbl) ;
      Out_channel.newline stdout;
      Attrs.print (snd block);
      Out_channel.newline stdout;
      print_s ([%sexp_of:Bril.value_expr list] exprs_to_delete);
      Out_channel.newline stdout;
      delete_computations exprs_to_delete block)
    graph

let insert_computations exprs graph =
  let _ = exprs in
  CFG.iter_edges_e
    (fun ((s, _), attrs, (d, _)) ->
      let exprs_to_insert =
        interp_bitv exprs @@ Attrs.get attrs "insert"
      in
      Printf.printf "%s -> %s\n" (Ident.string_of_lbl s.lbl) (Ident.string_of_lbl d.lbl);
      Attrs.print attrs;
      print_s ([%sexp_of:Bril.value_expr list] exprs_to_insert);
      Out_channel.newline stdout)
    graph;
  graph
