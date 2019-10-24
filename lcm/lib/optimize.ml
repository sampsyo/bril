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

let make_computation expr_typs exprs expr =
  let loc = expr_loc exprs expr in
  ValueInstr { op = expr; dest = loc; typ = Map.find_exn expr_typs expr }

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

let delete_computations_in exprs_to_delete block =
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
      delete_computations_in exprs_to_delete block)
    graph

let replace_block old_block new_block graph =
  CFG.map_vertex
    (fun b ->
      if (fst b).lbl = (fst old_block).lbl
      then new_block
      else b)
  graph

let insert_computations_at expr_typs exprs exprs_to_compute edge graph =
  let (src, _, dst) = edge in
  let instrs = List.map ~f:(make_computation expr_typs exprs) exprs_to_compute in
  let new_block = 
    { lbl = Ident.fresh_lbl "lcm_inserted_block";
      body = instrs;
      term = Jmp (fst dst).lbl}
  in
  let new_vtx = new_block, Attrs.create () in
  let out_edge = (new_vtx, Attrs.create (), dst) in
  let graph = CFG.remove_edge_e graph edge in
  let graph = CFG.add_vertex graph new_vtx in
  let graph = CFG.add_edge_e graph out_edge in
  let fix_lbl l = if l = (fst dst).lbl then new_block.lbl else l in
  let src_term' : Bril.term_op =
    match (fst src).term with
    | Ret -> failwith "edge out of returning node???"
    | Jmp l -> Jmp (fix_lbl l)
    | Br { cond; true_lbl; false_lbl } ->
       Br { cond = cond;
            true_lbl = fix_lbl true_lbl;
            false_lbl = fix_lbl false_lbl }
  in
  let src' = { (fst src) with term = src_term' }, snd src in
  let in_edge = (src', Attrs.create (), new_vtx) in
  let graph = replace_block src src' graph in
  CFG.add_edge_e graph in_edge

let insert_computations expr_typs exprs graph =
  CFG.fold_edges_e
    (fun edge graph' ->
      let (_, attrs, _) = edge in
      let exprs_to_insert =
        interp_bitv exprs @@ Attrs.get attrs "insert"
      in
      if List.length exprs_to_insert > 0
      then insert_computations_at expr_typs exprs exprs_to_insert edge graph'
      else graph')
    graph graph;
