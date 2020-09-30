open! Core
open! Common

type label = string [@@deriving compare, equal, sexp_of]
type arg = string [@@deriving compare, equal, sexp_of]

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
[@@deriving compare, equal, sexp_of]

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
    ( match json |> member "op" |> to_string with
    | "const" ->
      let const =
        match json |> member "type" |> Bril_type.of_json with
        | IntType -> Const.Int (json |> member "value" |> to_int)
        | BoolType -> Const.Bool (json |> member "value" |> to_bool)
      in
      Const (dest (), const)
    | op when Op.Binary.is_op op -> Binary (dest (), Op.Binary.of_string op, arg 0, arg 1)
    | op when Op.Unary.is_op op -> Unary (dest (), Op.Unary.of_string op, arg 0)
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
      ( [ ("op", `String (Op.Binary.to_string op)); ("args", `List [ `String arg1; `String arg2 ]) ]
      @ dest_to_json dest )
  | Unary (dest, op, arg) ->
    `Assoc
      ( [ ("op", `String (Op.Unary.to_string op)); ("args", `List [ `String arg ]) ]
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

let to_string =
  let dest_to_string (name, bril_type) = sprintf "%s: %s =" name (Bril_type.to_string bril_type) in
  function
  | Label label -> sprintf ".%s" label
  | Const (dest, const) -> sprintf "%s const %s" (dest_to_string dest) (Const.to_string const)
  | Binary (dest, op, arg1, arg2) ->
    sprintf "%s %s %s %s" (dest_to_string dest) (Op.Binary.to_string op) arg1 arg2
  | Unary (dest, op, arg) -> sprintf "%s %s %s" (dest_to_string dest) (Op.Unary.to_string op) arg
  | Jmp label -> sprintf "jmp %s" label
  | Br (arg, l1, l2) -> sprintf "br %s %s %s" arg l1 l2
  | Call (dest, func_name, args) ->
    List.filter
      ([ Option.value_map dest ~default:"" ~f:dest_to_string; func_name ] @ args)
      ~f:(Fn.non String.is_empty)
    |> String.concat ~sep:" "
  | Ret arg ->
    ( match arg with
    | Some arg -> sprintf "ret %s" arg
    | None -> "ret" )
  | Print args -> String.concat ~sep:" " ("print" :: args)
  | Nop -> "nop"
