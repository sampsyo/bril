open! Core
open! Common
module Bril_type = Bril_type
module Func = Func
module Instr = Instr
module Op = Op
include Common

type t = Func.t list [@@deriving sexp_of]

let from_json json =
  let open Yojson.Basic.Util in
  json |> member "functions" |> to_list_nonnull |> List.map ~f:Func.of_json

let to_json t = `Assoc [ ("functions", `List (List.map t ~f:Func.to_json)) ]
