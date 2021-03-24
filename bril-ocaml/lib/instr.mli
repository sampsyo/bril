open! Core
open! Common

type label = string [@@deriving compare, equal, sexp_of]
type arg = string [@@deriving compare, equal, sexp_of]

type t =
  | Label of label
  | Const of Dest.t * Const.t
  | Binary of Dest.t * Op.Binary.t * arg * arg
  | Unary of Dest.t * Op.Unary.t * arg
  | Jmp of label
  | Br of arg * label * label
  | Call of Dest.t option * string * arg list
  | Ret of arg option
  | Print of arg list
  | Nop
  | Phi of Dest.t * (label * arg) list
  | Speculate
  | Commit
  | Guard of arg * label
  | Alloc of (Dest.t * arg)
  | Free of arg
  | Store of (arg * arg)
  | Load of (Dest.t * arg)
  | PtrAdd of (Dest.t * arg * arg)
[@@deriving compare, equal, sexp_of]

val dest : t -> Dest.t option
val set_dest : Dest.t option -> t -> t
val args : t -> arg list
val set_args : arg list -> t -> t
val of_json : Yojson.Basic.t -> t
val to_json : t -> Yojson.Basic.t
val to_string : t -> string
