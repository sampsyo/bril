open! Core
open! Common
module Bril_type = Bril_type
module Const = Const
module Dest = Dest
module Func = Func
module Instr = Instr
module Op = Op

type t = Func.t list [@@deriving compare, equal, sexp_of]

let from_json json =
  let open Yojson.Basic.Util in
  json |> member "functions" |> to_list_nonnull |> List.map ~f:Func.of_json

let to_json t = `Assoc [ ("functions", `List (List.map t ~f:Func.to_json)) ]
let to_string t = t |> List.map ~f:Func.to_string |> String.concat ~sep:"\n\n"
