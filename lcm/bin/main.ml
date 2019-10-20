open Core
open Lcm

let fu _ =
  failwith "unimplemented"

let go dot file () =
  let prog = Parser.parse_bril file in
  let fn = List.hd_exn prog in
  let graph = Cfgify.cfgify_dirs fn.body in
  Cfg.dump_to_dot graph dot

let open_in_opt = function
  | Some path -> In_channel.create path
  | None -> In_channel.stdin

let open_out_opt = function
  | Some path -> Out_channel.create path
  | None -> Out_channel.stdout

let command =
  let spec =
    let open Command.Spec in
    empty
    +> flag "-dot" (optional string) ~doc:"<file.dot> Output CFG to .dot file"
    +> anon (maybe ("brilfile" %:string))
  in
  Command.basic_spec
    ~summary:"lcm: Lazy code motion for the Bril intermediate language"
    spec
    (fun dot_path bril_path ->
      let bril = open_in_opt bril_path in
      let dot = open_out_opt dot_path in
      go dot bril)

let () =
  Command.run ~version:"0.1.1" command
