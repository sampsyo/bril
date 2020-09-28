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
  [@@deriving compare, equal, sexp_of]

  val is_op : string -> bool
  val of_string : string -> t
  val to_string : t -> string
  val fold : t -> Const.t -> Const.t -> Const.t
end

module Unary : sig
  type t =
    | Not
    | Id
  [@@deriving compare, equal, sexp_of]

  val is_op : string -> bool
  val of_string : string -> t
  val to_string : t -> string
  val fold : t -> Const.t -> Const.t
end
