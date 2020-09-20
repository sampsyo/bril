open! Core
open! Common

type t =
  | Label of label
  | Const of dest * const
  | Binary of dest * Op.Binary.t * arg * arg
  | Unary of dest * Op.Unary.t * arg
  | Jmp of label
  | Br of arg * label * label
  | Call of dest option * func_name * arg list
  | Ret of arg option
  | Print of arg list
  | Nop
[@@deriving compare, sexp_of]

val dest : t -> dest option
val args : t -> arg list
val of_json : Yojson.Basic.t -> t
val to_json : t -> Yojson.Basic.t
