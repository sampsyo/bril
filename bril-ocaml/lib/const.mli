open! Core
open! Common

type t =
  | Int of int
  | Bool of bool
[@@deriving compare, sexp_of]
