open! Core
open! Common

type t = string * Bril_type.t [@@deriving compare, sexp_of]
