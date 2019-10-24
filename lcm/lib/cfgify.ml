open Core

exception BlockifyError of string
exception CfgifyError of string

let cfg_instr_of_instr (instr: Bril.instruction) : Cfg.instruction =
  match instr with
  | Nop -> Nop
  | ValueInstr v -> ValueInstr v
  | ConstInstr c -> ConstInstr c
  | EffectInstr (Print args) -> Print args
  | EffectInstr (TermOp _) -> 
     raise (BlockifyError "tried to put a terminating instruction in the middle of a basic block?")

let blockify (instrs: Bril.directive list) =
  let open Cfg in
  let open Bril in
  let rec blockify' instrs cur_block =
    match instrs with
    | Label lbl :: rest ->
       continue cur_block rest (Some lbl)
    | Instruction i :: rest ->
       begin match i with
       | EffectInstr (TermOp term) ->
         let cur_block = { cur_block with pre_term = Some term } in
         continue cur_block rest None
       | i ->
          let cur_block =
            { cur_block with
              pre_body = cfg_instr_of_instr i :: cur_block.pre_body }
          in
          blockify' rest cur_block
       end
    | [] ->
       if List.length cur_block.pre_body > 0 || cur_block.pre_term <> None
       then [cur_block]
       else []
  and continue cur_block instrs lbl =
    let cur_block =
      { cur_block with pre_body = List.rev cur_block.pre_body }
    in
    let next_block =
      { pre_lbl = lbl; pre_body = []; pre_term = None }
    in
    if List.length cur_block.pre_body > 0 || cur_block.pre_term <> None
    then cur_block :: blockify' instrs next_block
    else blockify' instrs next_block
  in
  let empty_block =
    { pre_lbl = None; pre_body = []; pre_term = None } 
  in
  blockify' instrs empty_block

let ensure_labeled (block: Cfg.pre_basic_block) : Cfg.pre_basic_block * Ident.lbl =
  match block.pre_lbl with
  | None ->
     let lbl = Ident.fresh_lbl "block" in
     { block with pre_lbl = Some lbl }, lbl
  | Some lbl ->
     block, lbl

let normalize_blocks (pre_blocks: Cfg.pre_basic_block list) : Cfg.basic_block list =
  let open Cfg in
  let rec normalize' pre_blocks blocks =
    match pre_blocks, blocks with
    | [], _ -> blocks
    | last_pre_block :: pre_blocks, [] ->
       let last_pre_block, lbl = ensure_labeled last_pre_block in
       let last_block =
         match last_pre_block.pre_term with
         | None ->
            { lbl = lbl; body = last_pre_block.pre_body; term = Ret }
         | Some op ->
            { lbl = lbl; body = last_pre_block.pre_body; term = op }
       in
       normalize' pre_blocks [last_block]
    | pre_block :: pre_blocks, last_block :: _ ->
       let pre_block, lbl = ensure_labeled pre_block in
       let block =
         match pre_block.pre_term with
         | None ->
            { lbl = lbl; body = pre_block.pre_body; term = Jmp (last_block.lbl) }
         | Some op ->
            { lbl = lbl; body = pre_block.pre_body; term = op }
       in
       normalize' pre_blocks (block :: blocks)
  in
  normalize' (List.rev pre_blocks) []

let successors block =
  let open Cfg in
  match block.term with
  | Ret -> []
  | Jmp lbl -> [lbl]
  | Br { true_lbl; false_lbl; cond = _ } -> [true_lbl; false_lbl]

let add_edges map graph block =
  let add_edge graph lbl =
    let dst_block = Cfg.LabelMap.find_exn map lbl in
    Cfg.CFG.add_edge_e graph (block, Cfg.Attrs.create (), dst_block)
  in
  List.fold ~f:add_edge ~init:graph (successors (fst block))

let add_all_edges map graph blocks =
  List.fold ~f:(add_edges map) ~init:graph blocks

let cfgify_dirs (dirs : Bril.directive list) =
  let pre_blocks = blockify dirs in
  let blocks = normalize_blocks pre_blocks in
  let attr is_entry (b : Cfg.basic_block) =
    let a = Cfg.Attrs.create () in
    (* for debugging *)
    let key = "cfg_default_attrs_" ^ (Ident.string_of_lbl b.lbl) in
    let one = Bitv.init 1 (fun _ -> true) in
    let zero = Bitv.init 1 (fun _ -> false) in
    Hashtbl.set ~key:key ~data:one a;
    if is_entry
    then Hashtbl.set ~key:"entry" ~data:one a
    else Hashtbl.set ~key:"entry" ~data:zero a;
    a
  in
  let blocks =
    match blocks with
    | [] -> []
    | entry :: rest ->
       let entry = entry, attr true entry in
       let rest = List.map ~f:(fun b -> b, attr false b) rest in
       entry :: rest
  in
  let graph, map = List.fold ~init:Cfg.empty ~f:Cfg.add_block blocks in
  add_all_edges map graph blocks
