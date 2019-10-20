open Core
open Lcm

let fu _ =
  failwith "unimplemented"

let go dot file () =
  let prog = Parser.parse_bril file in
  let fn = List.hd_exn prog in
  let graph = Cfgify.cfgify_dirs fn.body in
  Cfg.dump_to_dot graph dot;
  let transp block = Analyze.transparent Analyze.expression block in
  let computes block = Analyze.computes Analyze.expression block in
  let anticipates block = Analyze.anticipates Analyze.expression block in
  let avail_init b =
    match Cfg.CFG.pred graph b with
    | [] -> false
    | _ -> true
  in
  let ant_init b =
    match Cfg.CFG.succ graph b with
    | [] -> false
    | _ -> true
  in
  let avail = Analyze.Availability.analyze avail_init graph in
  let ant = Analyze.Anticipatability.analyze ant_init graph in
  Cfg.CFG.iter_vertex (fun v ->
      print_string (Ident.string_of_lbl v.lbl);
      print_string "\n\t";
      print_string @@ string_of_bool (transp v);
      print_string "\t";
      print_string @@ string_of_bool (computes v);
      print_string "\t";
      print_string @@ string_of_bool (anticipates v);
      print_string "\t";
      print_string @@ string_of_bool (avail v);
      print_string "\t";
      print_string @@ string_of_bool (ant v);
      print_string "\n")
    graph

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
