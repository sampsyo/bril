open Core
open Lcm

let fu _ =
  failwith "unimplemented"

let go dot file () =
  let prog = Parser.parse_bril file in
  let fn = List.hd_exn prog in
  let graph = Cfgify.cfgify_dirs fn.body in
  let expr_locs = Analyze.aggregate_expression_locs graph in
  let expressions = Analyze.ExprMap.keys expr_locs in
  let graph = Optimize.unify_expression_locations expressions graph in
  let module Exprs = struct
      let expressions = expressions
      let len = List.length expressions
      let build ~f =
        let bit_at i =
          let expr = List.nth_exn expressions i in
          f expr
        in
        Bitv.init len bit_at
    end
  in
  let module Analyze = Analyze.Analyze(Exprs) in
  let graph =
    Analyze.Entry.run graph
    |> Analyze.Exit.run
    |> Analyze.Transparent.run
    |> Analyze.Computes.run
    |> Analyze.LocallyAnticipates.run
    |> Analyze.AvailabilityIn.run
    |> Analyze.AvailabilityOut.run
    |> Analyze.AnticipatabilityOut.run
    |> Analyze.AnticipatabilityIn.run
    |> Analyze.Earliest.run
    |> Analyze.Later.run
    |> Analyze.Insert.run
    |> Analyze.Delete.run
    |> Optimize.delete_computations expressions
    |> Optimize.insert_computations expressions
  in
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
