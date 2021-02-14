
open! Core

let main channel =
  let funcs = Yojson.Basic.from_channel channel |> Bril.from_json in
  let func_map =
    List.fold funcs ~init:String.Map.empty
      ~f:(fun acc func -> String.Map.add_exn acc ~key:func.name ~data:func) in
  List.iter ~f:(Type_check.perform func_map) funcs

let () =
  match Sys.get_argv () with
  | [| _ |] -> main In_channel.stdin
  | [| _; fname |] -> In_channel.create fname |> main
  | _ -> prerr_endline "Unexpected argument"
