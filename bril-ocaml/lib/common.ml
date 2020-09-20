open! Core

type const =
  | Int of int
  | Bool of bool
[@@deriving compare, sexp_of]

type dest = string * Bril_type.t [@@deriving compare, sexp_of]
type label = string [@@deriving compare, sexp_of]
type arg = string [@@deriving compare, sexp_of]
type func_name = string [@@deriving compare, sexp_of]

let has_key json key =
  let open Yojson.Basic.Util in
  match json |> member key with
  | `Null -> false
  | _ -> true

let to_list_nonnull =
  let open Yojson.Basic.Util in
  function
  | `Null -> []
  | json -> to_list json
