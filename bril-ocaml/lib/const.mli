open! Core
open! Common

type t =
  | Int of int
  | Bool of bool
  | Float of float
[@@deriving compare, equal, sexp_of]

val to_string : t -> string
