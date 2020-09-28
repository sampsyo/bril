open! Core

type t =
  | IntType
  | BoolType
[@@deriving compare, equal, sexp_of]

let of_json =
  let open Yojson.Basic.Util in
  function
  | `String "int" -> IntType
  | `String "bool" -> BoolType
  | json -> failwithf "invalid type: %s" (json |> to_string) ()

let of_json_opt = function
  | `Null -> None
  | json -> Some (of_json json)

let to_json = function
  | IntType -> `String "int"
  | BoolType -> `String "bool"

let to_string = function
  | IntType -> "int"
  | BoolType -> "bool"
