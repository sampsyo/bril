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
  | Phi of Dest.t * (label * arg) list
  | Speculate
  | Commit
  | Guard of arg * label
  | Alloc of (Dest.t * arg)
  | Free of arg
  | Store of (arg * arg)
  | Load of (Dest.t * arg)
  | PtrAdd of (Dest.t * arg * arg)
[@@deriving compare, equal, sexp_of]

let to_string =
  let dest_to_string (name, bril_type) = sprintf "%s: %s =" name (Bril_type.to_string bril_type) in
  function
  | Label label -> sprintf ".%s" label
  | Const (dest, const) -> sprintf "%s const %s" (dest_to_string dest) (Const.to_string const)
  | Binary (dest, op, arg1, arg2) ->
    sprintf "%s %s %s %s" (dest_to_string dest) (Op.Binary.to_string op) arg1 arg2
  | Unary (dest, op, arg) -> sprintf "%s %s %s" (dest_to_string dest) (Op.Unary.to_string op) arg
  | Jmp label -> sprintf "jmp .%s" label
  | Br (arg, l1, l2) -> sprintf "br %s .%s .%s" arg l1 l2
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
  | Phi (dest, alist) ->
    sprintf
      "%s phi %s"
      (dest_to_string dest)
      (List.map alist ~f:(fun (label, arg) -> sprintf ".%s %s" label arg) |> String.concat ~sep:" ")
  | Speculate -> "speculate"
  | Commit -> "commit"
  | Guard (arg, l) -> sprintf "guard %s .%s" arg l
  | Alloc (dst, arg) -> sprintf "%s alloc %s" (dest_to_string dst) arg
  | Store (arg1, arg2) -> sprintf "store %s %s" arg1 arg2
  | Load (dst, arg) -> sprintf "%s load %s" (dest_to_string dst) arg
  | PtrAdd (dst, arg1, arg2) -> sprintf "%s ptradd %s %s" (dest_to_string dst) arg1 arg2
  | Free arg -> sprintf "free %s" arg

let dest = function
  | Const (dest, _)
  | Binary (dest, _, _, _)
  | Unary (dest, _, _)
  | Phi (dest, _)
  | Alloc (dest, _)
  | PtrAdd (dest, _, _)
  | Load (dest, _) ->
    Some dest
  | Call (dest, _, _) -> dest
  | Nop
  | Speculate
  | Commit
  | Label _
  | Jmp _
  | Br (_, _, _)
  | Ret _
  | Print _
  | Guard (_, _)
  | Free _
  | Store _ ->
    None

let set_dest dest t =
  match (t, dest) with
  | (Const (_, const), Some dest) -> Const (dest, const)
  | (Binary (_, op, arg1, arg2), Some dest) -> Binary (dest, op, arg1, arg2)
  | (Unary (_, op, arg), Some dest) -> Unary (dest, op, arg)
  | (Call (_, f, args), dest) -> Call (dest, f, args)
  | (Phi (_, params), Some dest) -> Phi (dest, params)
  | (Alloc (_, arg), Some dest) -> Alloc (dest, arg)
  | (Load (_, arg), Some dest) -> Load (dest, arg)
  | (PtrAdd (_, a1, a2), Some dest) -> PtrAdd (dest, a1, a2)
  | (instr, None) -> instr
  | _ ->
    let dest = [%sexp_of: Dest.t Option.t] dest in
    failwithf !"Cannot [set_dest] on instruction %{to_string} with dest %{Sexp}" t dest ()

let args = function
  | Binary (_, _, arg1, arg2) -> [ arg1; arg2 ]
  | Unary (_, _, arg)
  | Br (arg, _, _)
  | Guard (arg, _) ->
    [ arg ]
  | Call (_, _, args)
  | Print args ->
    args
  | Alloc ((_ : Dest.t), arg) -> [ arg ]
  | Free arg -> [ arg ]
  | Store (arg1, arg2) -> [ arg1; arg2 ]
  | Load ((_ : Dest.t), arg) -> [ arg ]
  | PtrAdd ((_ : Dest.t), arg1, arg2) -> [ arg1; arg2 ]
  | Ret arg -> Option.value_map arg ~default:[] ~f:List.return
  | Phi ((_ : Dest.t), label_and_args) -> List.map label_and_args ~f:snd
  | (Nop | Speculate | Commit | Label _ | Const (_, _) | Jmp _) as instr ->
    failwithf "Cannot call [args] on %s" (to_string instr) ()

let set_args args t =
  match (t, args) with
  | (Binary (dest, op, _, _), [ arg1; arg2 ]) -> Binary (dest, op, arg1, arg2)
  | (Unary (dest, op, _), [ arg ]) -> Unary (dest, op, arg)
  | (Br (_, l1, l2), [ arg ]) -> Br (arg, l1, l2)
  | (Call (dest, f, _), args) -> Call (dest, f, args)
  | (Print _, args) -> Print args
  | (Ret _, []) -> Ret None
  | (Ret _, [ arg ]) -> Ret (Some arg)
  | (Guard (_, l), [ arg ]) -> Guard (arg, l)
  | (Alloc (dst, _), [ arg ]) -> Alloc (dst, arg)
  | (Free _, [ arg ]) -> Free arg
  | (Store (_, _), [ a1; a2 ]) -> Store (a1, a2)
  | (Load (dst, _), [ a ]) -> Load (dst, a)
  | (PtrAdd (dst, _, _), [ a1; a2 ]) -> PtrAdd (dst, a1, a2)
  | (instr, []) -> instr
  | _ -> failwith "invalid set_args"

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
        | PtrType _ -> failwith "pointer is not supported in constants"
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
    | "phi" -> Phi (dest (), List.zip_exn (labels ()) (args ()))
    | "speculate" -> Speculate
    | "commit" -> Commit
    | "guard" -> Guard (arg 0, label 0)
    | "alloc" -> Alloc (dest (), arg 0)
    | "free" -> Free (arg 0)
    | "store" -> Store (arg 0, arg 1)
    | "load" -> Load (dest (), arg 0)
    | "ptradd" -> PtrAdd (dest (), arg 0, arg 1)
    | op -> failwithf "invalid op: %s" op () )
  | json -> failwithf "invalid label: %s" (json |> to_string) ()

let to_json =
  let dest_to_json (name, bril_type) =
    [ ("dest", `String name); ("type", Bril_type.to_json bril_type) ]
  in
  let build_op ?dest ?args ~op () =
    `Assoc
      ( [ ("op", `String op) ]
      @ ( match args with
        | None -> []
        | Some args -> [ ("args", `List (List.map args ~f:(fun a -> `String a))) ] )
      @
      match dest with
      | None -> []
      | Some dest -> dest_to_json dest )
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
  | Phi (dest, alist) ->
    `Assoc
      ( [
          ("op", `String "phi");
          ("labels", `List (List.map alist ~f:(fun (label, _) -> `String label)));
          ("args", `List (List.map alist ~f:(fun (_, arg) -> `String arg)));
        ]
      @ dest_to_json dest )
  | Speculate -> `Assoc [ ("op", `String "speculate") ]
  | Commit -> `Assoc [ ("op", `String "commit") ]
  | Guard (arg, l) ->
    `Assoc
      [ ("op", `String "guard"); ("args", `List [ `String arg ]); ("labels", `List [ `String l ]) ]
  | Alloc (dest, arg) -> build_op ~op:"alloc" ~args:[ arg ] ~dest ()
  | Free arg -> build_op ~op:"free" ~args:[ arg ] ()
  | Load (dest, arg) -> build_op ~op:"load" ~args:[ arg ] ~dest ()
  | Store (arg1, arg2) -> build_op ~op:"load" ~args:[ arg1; arg2 ] ()
  | PtrAdd (dest, arg1, arg2) -> build_op ~op:"ptradd" ~args:[ arg1; arg2 ] ~dest ()
