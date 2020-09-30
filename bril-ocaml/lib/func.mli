open! Core
open! Common

type t = {
  name : string;
  args : Dest.t list;
  ret_type : Bril_type.t option;
  blocks : Instr.t list String.Map.t;
  order : string list;
  preds : string list String.Map.t;
  succs : string list String.Map.t;
}
[@@deriving compare, equal, sexp_of]

val instrs : t -> Instr.t list
val set_instrs : t -> Instr.t list -> t
val of_json : Yojson.Basic.t -> t
val to_json : t -> Yojson.Basic.t
val to_string : t -> string
