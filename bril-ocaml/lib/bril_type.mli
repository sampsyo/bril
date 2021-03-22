open! Core

type t =
  | IntType
  | BoolType
  | PtrType of t
[@@deriving compare, equal, sexp_of]

val of_json : Yojson.Basic.t -> t
val of_json_opt : Yojson.Basic.t -> t option
val to_json : t -> Yojson.Basic.t
val to_string : t -> string
