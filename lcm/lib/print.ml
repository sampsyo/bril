open Core
open Cfg

let find_entry graph =
  let found =
    CFG.fold_vertex (fun v found ->
        Printf.printf "\n%s\n" (Ident.string_of_lbl (fst v).lbl);
        match Hashtbl.find (snd v) "entry" with
        | Some e ->
           if not (Bitv.all_zeros e)
           then Some v
           else found
        | None -> found)
      graph None
  in
  match found with
  | Some v -> v
  | None -> failwith "no entry node..?"

let collect_lbl l : Bril.directive =
  Label l

let collect_instruction : Cfg.instruction -> Bril.instruction =
  function
  | Print args -> EffectInstr (Print args)
  | ValueInstr v -> ValueInstr v
  | ConstInstr c -> ConstInstr c
  | Nop -> Nop

let collect_instructions : Cfg.instruction list -> Bril.directive list =
  List.map ~f:(fun i -> Bril.Instruction (collect_instruction i))

let collect_term (t : Bril.term_op) =
  Bril.Instruction (EffectInstr (TermOp t))

let collect_block (block, _) =
  collect_lbl block.lbl ::
  collect_instructions block.body @
  [collect_term block.term]

let rec collect_blocks graph seen blocks: Bril.directive list =
  match blocks with
  | [] -> []
  | block :: rest ->
     let l = (fst block).lbl in
     if List.mem seen l
          ~equal:(fun x y -> Ident.cmp_lbl x y = 0)
     then collect_blocks graph seen rest
     else let succs = CFG.succ graph block in
          let blocks = succs @ rest in
          collect_block block @ collect_blocks graph (l :: seen) blocks
     
let decfg graph : Bril.program =
  let entry = find_entry graph in
  [{name = Ident.fn_of_string "main";
    body = collect_blocks graph [] [entry]}]
