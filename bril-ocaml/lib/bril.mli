open! Core
open! Common
module Bril_type = Bril_type
module Const = Const
module Dest = Dest
module Func = Func
module Instr = Instr
module Op = Op

type t = Func.t list [@@deriving compare, equal, sexp_of]

val from_json : Yojson.Basic.t -> t
val to_json : t -> Yojson.Basic.t
val to_string : t -> string
