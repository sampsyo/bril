open Core

exception BlockifyError of string

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
       let cur_block =
         { cur_block with
           pre_body = cfg_instr_of_instr i :: cur_block.pre_body }
       in
       begin match i with
       | EffectInstr (TermOp term) ->
         let cur_block = { cur_block with pre_term = Some term } in
         continue cur_block rest None
       | _ -> blockify' rest cur_block
       end
    | [] -> []
  and continue cur_block instrs lbl =
    let cur_block =
      { cur_block with pre_body = List.rev cur_block.pre_body }
    in
    let next_block =
      { pre_lbl = lbl; pre_body = []; pre_term = None }
    in
    if List.length cur_block.pre_body > 0
    then cur_block :: blockify' instrs next_block
    else blockify' instrs next_block
  in
  let empty_block =
    { pre_lbl = None; pre_body = []; pre_term = None } 
  in
  List.rev @@ blockify' instrs empty_block
