open! Core
open! Common

type t = {
  name : string;
  args : Dest.t list;
  ret_type : Bril_type.t option;
  blocks : Instr.t list String.Map.t;
  order : string list;
  preds : string list String.Map.t;
  succs : string list String.Map.t;
}
[@@deriving compare, equal, sexp_of]

let instrs { blocks; order; _ } = List.concat_map order ~f:(Map.find_exn blocks)

let process_instrs instrs =
  let block_name i = sprintf "block_%d" i in
  let (name, block, blocks) =
    List.fold
      instrs
      ~init:(block_name 0, [], [])
      ~f:(fun (name, block, blocks) (instr : Instr.t) ->
        match instr with
        | Label label ->
          if List.is_empty block then ("label_" ^ label, [ instr ], blocks)
          else ("label_" ^ label, [ instr ], (name, List.rev block) :: blocks)
        | Jmp _
        | Br _
        | Ret _ ->
          let blocks = (name, List.rev (instr :: block)) :: blocks in
          (block_name (List.length blocks), [], blocks)
        | _ -> (name, instr :: block, blocks))
  in
  let blocks =
    (name, List.rev block) :: blocks
    |> List.rev_filter ~f:(fun (_, block) -> not (List.is_empty block))
  in
  let order = List.map blocks ~f:fst in
  let succs =
    List.mapi blocks ~f:(fun i (name, block) ->
        let next =
          match List.last_exn block with
          | Jmp label -> [ "label_" ^ label ]
          | Br (_, l1, l2) -> [ "label_" ^ l1; "label_" ^ l2 ]
          | Ret _ -> []
          | _ ->
            ( match List.nth blocks (i + 1) with
            | None -> []
            | Some (name, _) -> [ name ] )
        in
        (name, next))
    |> String.Map.of_alist_exn
  in
  let preds =
    Map.fold
      succs
      ~init:(List.map order ~f:(fun name -> (name, [])) |> String.Map.of_alist_exn)
      ~f:(fun ~key:name ~data:succs preds ->
        List.fold succs ~init:preds ~f:(fun preds succ -> Map.add_multi preds ~key:succ ~data:name))
  in
  (String.Map.of_alist_exn blocks, order, preds, succs)

let set_instrs t instrs =
  let (blocks, order, preds, succs) = process_instrs instrs in
  { t with blocks; order; preds; succs }

let of_json json =
  let open Yojson.Basic.Util in
  let arg_of_json json =
    (json |> member "name" |> to_string, json |> member "type" |> Bril_type.of_json)
  in
  let name = json |> member "name" |> to_string in
  let args = json |> member "args" |> to_list_nonnull |> List.map ~f:arg_of_json in
  let ret_type = json |> member "type" |> Bril_type.of_json_opt in
  let instrs = json |> member "instrs" |> to_list_nonnull |> List.map ~f:Instr.of_json in
  let (blocks, order, preds, succs) = process_instrs instrs in
  { name; args; ret_type; blocks; order; preds; succs }

let to_json t =
  `Assoc
    ( [
        ("name", `String t.name);
        ( "args",
          `List
            (List.map t.args ~f:(fun (name, bril_type) ->
                 `Assoc [ ("name", `String name); ("type", Bril_type.to_json bril_type) ])) );
        ("instrs", `List (instrs t |> List.map ~f:Instr.to_json));
      ]
    @ Option.value_map t.ret_type ~default:[] ~f:(fun t -> [ ("type", Bril_type.to_json t) ]) )

let to_string { name; args; ret_type; blocks; order; _ } =
  let header =
    sprintf
      "%s%s%s {"
      name
      ( match args with
      | [] -> ""
      | args ->
        sprintf
          "(%s)"
          ( List.map args ~f:(fun (name, bril_type) ->
                sprintf "%s: %s" name (Bril_type.to_string bril_type))
          |> String.concat ~sep:", " ) )
      (Option.value_map ret_type ~default:"" ~f:Bril_type.to_string)
  in
  let body =
    order
    |> List.concat_map ~f:(Map.find_exn blocks)
    |> List.map ~f:Instr.to_string
    |> String.concat ~sep:";\n"
  in
  sprintf "%s\n%s\n}" header body
