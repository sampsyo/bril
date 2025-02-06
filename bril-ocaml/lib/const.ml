open! Core
open! Common

type t =
  | Int of int
  | Bool of bool
  | Float of float
[@@deriving compare, equal, sexp_of]

let to_string = function
  | Int i -> Int.to_string i
  | Bool b -> Bool.to_string b
  | Float f -> Float.to_string f
