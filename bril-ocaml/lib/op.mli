open! Core

module Binary : sig
  type t =
    | Add
    | Mul
    | Sub
    | Div
    | Eq
    | Lt
    | Gt
    | Le
    | Ge
    | And
    | Or
  [@@deriving sexp_of, equal]

  val by_name : (string * t) list
  val by_op : (t * string) list
end

module Unary : sig
  type t =
    | Not
    | Id
  [@@deriving sexp_of, equal]

  val by_name : (string * t) list
  val by_op : (t * string) list
end
