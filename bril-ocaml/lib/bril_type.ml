open! Core

type t =
  | IntType
  | BoolType
  | SumType of t list
  | ProdType of t list
[@@deriving compare, equal, sexp_of]

let rec of_json =
  let open Yojson.Basic.Util in
  function
  | `String "int" -> IntType
  | `String "bool" -> BoolType
  | `Assoc [("sum", `List ts)] -> SumType (List.map ~f:of_json ts)
  | `Assoc [("product", `List ts)] -> ProdType (List.map ~f:of_json ts)
  | json -> failwithf "invalid type: %s" (json |> to_string) ()

let of_json_opt = function
  | `Null -> None
  | json -> Some (of_json json)

let rec to_json = function
  | IntType -> `String "int"
  | BoolType -> `String "bool"
  | SumType ts -> `Assoc [("sum", `List (List.map ~f:to_json ts))]
  | ProdType ts -> `Assoc [("product", `List (List.map ~f:to_json ts))]

let rec to_string = function
  | IntType -> "int"
  | BoolType -> "bool"
  | SumType ts -> "sum<" ^ String.concat ~sep:", " (List.map ~f:to_string ts) ^ ">"
  | ProdType ts -> "product<" ^ String.concat ~sep:", " (List.map ~f:to_string ts) ^ ">"
