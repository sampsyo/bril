open! Core

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
