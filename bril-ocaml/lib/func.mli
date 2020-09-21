open! Core
open! Common

type t = {
  name : string;
  args : Dest.t list;
  ret_type : Bril_type.t option;
  blocks : Instr.t list String.Map.t;
  order : string list;
  cfg : string list String.Map.t;
}
[@@deriving compare, sexp_of]

val instrs : t -> Instr.t list
val of_json : Yojson.Basic.t -> t
val to_json : t -> Yojson.Basic.t
