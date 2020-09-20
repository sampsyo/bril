open! Core
open! Common

type label = string [@@deriving compare, sexp_of]
type arg = string [@@deriving compare, sexp_of]

type t =
  | Label of label
  | Const of Dest.t * Const.t
  | Binary of Dest.t * Op.Binary.t * arg * arg
  | Unary of Dest.t * Op.Unary.t * arg
  | Jmp of label
  | Br of arg * label * label
  | Call of Dest.t option * string * arg list
  | Ret of arg option
  | Print of arg list
  | Nop
[@@deriving compare, sexp_of]

let dest = function
  | Const (dest, _)
  | Binary (dest, _, _, _)
  | Unary (dest, _, _) ->
    Some dest
  | Call (dest, _, _) -> dest
  | Label _
  | Jmp _
  | Br _
  | Ret _
  | Print _
  | Nop ->
    None

let args = function
  | Binary (_, _, arg1, arg2) -> [ arg1; arg2 ]
  | Unary (_, _, arg)
  | Br (arg, _, _) ->
    [ arg ]
  | Call (_, _, args)
  | Print args ->
    args
  | Ret arg -> Option.value_map arg ~default:[] ~f:List.return
  | Const _
  | Label _
  | Jmp _
  | Nop ->
    []

let of_json json =
  let open Yojson.Basic.Util in
  match json |> member "label" with
  | `String label -> Label label
  | `Null ->
    let dest () =
      (json |> member "dest" |> to_string, json |> member "type" |> Bril_type.of_json)
    in
    let args () = json |> member "args" |> to_list_nonnull |> List.map ~f:to_string in
    let labels () = json |> member "labels" |> to_list_nonnull |> List.map ~f:to_string in
    let arg = List.nth_exn (args ()) in
    let label = List.nth_exn (labels ()) in
    let mem = List.Assoc.mem ~equal:String.equal in
    let find = List.Assoc.find_exn ~equal:String.equal in
    ( match json |> member "op" |> to_string with
    | "const" ->
      let const =
        match json |> member "type" |> Bril_type.of_json with
        | IntType -> Const.Int (json |> member "value" |> to_int)
        | BoolType -> Const.Bool (json |> member "value" |> to_bool)
      in
      Const (dest (), const)
    | op when mem Op.Binary.by_name op -> Binary (dest (), find Op.Binary.by_name op, arg 0, arg 1)
    | op when mem Op.Unary.by_name op -> Unary (dest (), find Op.Unary.by_name op, arg 0)
    | "jmp" -> Jmp (label 0)
    | "br" -> Br (arg 0, label 0, label 1)
    | "call" ->
      Call
        ( (if has_key json "dest" then Some (dest ()) else None),
          json |> member "funcs" |> to_list_nonnull |> List.hd_exn |> to_string,
          args () )
    | "ret" -> Ret (if List.is_empty (args ()) then None else Some (arg 0))
    | "print" -> Print (args ())
    | "nop" -> Nop
    | op -> failwithf "invalid op: %s" op () )
  | json -> failwithf "invalid label: %s" (json |> to_string) ()

let to_json =
  let dest_to_json (name, bril_type) =
    [ ("dest", `String name); ("type", Bril_type.to_json bril_type) ]
  in
  function
  | Label label -> `Assoc [ ("label", `String label) ]
  | Const (dest, const) ->
    `Assoc
      ( [
          ("op", `String "const");
          ( "value",
            match const with
            | Int i -> `Int i
            | Bool b -> `Bool b );
        ]
      @ dest_to_json dest )
  | Binary (dest, op, arg1, arg2) ->
    `Assoc
      ( [
          ("op", `String Op.Binary.(List.Assoc.find_exn by_op op ~equal));
          ("args", `List [ `String arg1; `String arg2 ]);
        ]
      @ dest_to_json dest )
  | Unary (dest, op, arg) ->
    `Assoc
      ( [
          ("op", `String Op.Unary.(List.Assoc.find_exn by_op op ~equal));
          ("args", `List [ `String arg ]);
        ]
      @ dest_to_json dest )
  | Jmp label -> `Assoc [ ("op", `String "jmp"); ("labels", `List [ `String label ]) ]
  | Br (arg, l1, l2) ->
    `Assoc
      [
        ("op", `String "br");
        ("args", `List [ `String arg ]);
        ("labels", `List [ `String l1; `String l2 ]);
      ]
  | Call (dest, func_name, args) ->
    `Assoc
      ( [
          ("op", `String "call");
          ("funcs", `List [ `String func_name ]);
          ("args", `List (List.map args ~f:(fun arg -> `String arg)));
        ]
      @ Option.value_map dest ~default:[] ~f:dest_to_json )
  | Ret arg ->
    `Assoc
      [
        ("op", `String "ret");
        ("args", Option.value_map arg ~default:`Null ~f:(fun arg -> `List [ `String arg ]));
      ]
  | Print args ->
    `Assoc [ ("op", `String "print"); ("args", `List (List.map args ~f:(fun arg -> `String arg))) ]
  | Nop -> `Assoc [ ("op", `String "nop") ]
