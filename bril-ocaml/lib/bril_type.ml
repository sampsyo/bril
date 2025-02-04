open! Core

type t =
  | IntType
  | BoolType
  | FloatType
  | PtrType of t
[@@deriving compare, equal, sexp_of]

let rec of_json =
  let open Yojson.Basic.Util in
  function
  | `String "int" -> IntType
  | `String "bool" -> BoolType
  | `String "float" -> FloatType
  | `Assoc [ ("ptr", inner) ] -> PtrType (of_json inner)
  | json -> failwithf "invalid type: %s" (json |> to_string) ()

let of_json_opt = function
  | `Null -> None
  | json -> Some (of_json json)

let rec to_json = function
  | IntType -> `String "int"
  | BoolType -> `String "bool"
  | FloatType -> `String "float"
  | PtrType inner -> `Assoc [ ("ptr", to_json inner) ]

let rec to_string = function
  | IntType -> "int"
  | BoolType -> "bool"
  | FloatType -> "float"
  | PtrType inner -> sprintf "ptr[%s]" (to_string inner)
