open! Core

type bril_type =
  | IntType
  | BoolType
[@@deriving sexp_of]

type const =
  | Int of int
  | Bool of bool
[@@deriving sexp_of]

type dest = string * bril_type [@@deriving sexp_of]

type label = string [@@deriving sexp_of]

type arg = string [@@deriving sexp_of]

type func_name = string [@@deriving sexp_of]

type binop =
  | Add
  | Mul
  | Sub
  | Div
  | Eq
  | Lt
  | Gt
  | Le
  | Ge
  | Not
  | And
  | Or
[@@deriving sexp_of]

type unop =
  | Not
  | Id
[@@deriving sexp_of]

type instr =
  | Label of label
  | Const of dest * const
  | Binary of dest * binop * arg * arg
  | Unary of dest * unop * arg
  | Jmp of label
  | Br of arg * label * label
  | Call of dest option * func_name * arg list
  | Ret of arg option
  | Print of arg list
  | Nop
[@@deriving sexp_of]

type func = {
  name : func_name;
  args : dest list;
  ret_type : bril_type option;
  body : instr list;
}
[@@deriving sexp_of]

type t = { funcs : func list } [@@deriving sexp_of]

let parse input =
  let json = Yojson.Basic.from_string input in
  let open Yojson.Basic.Util in
  let has_key json key =
    match json |> member key with
    | `Null -> false
    | _ -> true
  in
  let to_list = function
    | `Null -> []
    | json -> to_list json
  in
  let to_type = function
    | `String "int" -> IntType
    | `String "bool" -> BoolType
    | json -> failwithf "invalid type: %s" (json |> to_string) ()
  in
  let to_type_option = function
    | `Null -> None
    | json -> Some (to_type json)
  in
  let to_arg json = (json |> member "name" |> to_string, json |> member "type" |> to_type) in
  let binops =
    String.Map.of_alist_exn
      [
        ("add", Add);
        ("mul", Mul);
        ("sub", Sub);
        ("div", Div);
        ("eq", Eq);
        ("lt", Lt);
        ("gt", Gt);
        ("le", Le);
        ("ge", Ge);
        ("not", Not);
        ("and", And);
        ("or", Or);
      ]
  in
  let unops = String.Map.of_alist_exn [ ("not", Not); ("id", Id) ] in
  let to_instr json =
    match json |> member "label" with
    | `String label -> Label label
    | `Null -> (
        let dest () = (json |> member "dest" |> to_string, json |> member "type" |> to_type) in
        let args () = json |> member "args" |> to_list |> List.map ~f:to_string in
        let labels () = json |> member "labels" |> to_list |> List.map ~f:to_string in
        let arg = List.nth_exn (args ()) in
        let label = List.nth_exn (labels ()) in
        match json |> member "op" |> to_string with
        | "const" ->
            let const =
              match json |> member "type" |> to_type with
              | IntType -> Int (json |> member "value" |> to_int)
              | BoolType -> Bool (json |> member "value" |> to_bool)
            in
            Const (dest (), const)
        | op when Map.mem binops op -> Binary (dest (), Map.find_exn binops op, arg 0, arg 1)
        | op when Map.mem unops op -> Unary (dest (), Map.find_exn unops op, arg 0)
        | "jmp" -> Jmp (label 0)
        | "br" -> Br (arg 0, label 0, label 1)
        | "call" ->
            Call
              ( (if has_key json "dest" then Some (dest ()) else None),
                json |> member "funcs" |> to_list |> List.hd_exn |> to_string,
                args () )
        | "ret" -> Ret (if List.is_empty (args ()) then None else Some (arg 0))
        | "print" -> Print (args ())
        | "nop" -> Nop
        | op -> failwithf "invalid op: %s" op () )
    | json -> failwithf "invalid label: %s" (json |> to_string) ()
  in
  let to_func json =
    let name = json |> member "name" |> to_string in
    let args = json |> member "args" |> to_list |> List.map ~f:to_arg in
    let ret_type = json |> member "type" |> to_type_option in
    let body = json |> member "instrs" |> to_list |> List.map ~f:to_instr in
    { name; args; ret_type; body }
  in
  { funcs = json |> member "functions" |> to_list |> List.map ~f:to_func }
