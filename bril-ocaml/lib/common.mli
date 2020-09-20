open! Core

type const =
  | Int of int
  | Bool of bool
[@@deriving compare, sexp_of]

type dest = string * Bril_type.t [@@deriving compare, sexp_of]
type label = string [@@deriving compare, sexp_of]
type arg = string [@@deriving compare, sexp_of]
type func_name = string [@@deriving compare, sexp_of]

val has_key : Yojson.Basic.t -> string -> bool
val to_list_nonnull : Yojson.Basic.t -> Yojson.Basic.t list
