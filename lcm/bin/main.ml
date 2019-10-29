open Core
open Lcm

let fu _ =
  failwith "unimplemented"

let go dot_before dot_after file () =
  let prog = Parser.parse_bril file in
  let fn = List.hd_exn prog in
  let graph = Cfgify.cfgify_dirs fn.body in
  (if Cfg.CFG.nb_vertex graph <= 1 then failwith "1 or fewer basic blocks, not optimizing!\n");
  begin match dot_before with
  | Some dot_before ->
     Cfg.dump_to_dot graph dot_before
  | None -> ()
  end;
  let expr_typs = Analyze.aggregate_expression_typs graph in
  let expressions = Analyze.ExprMap.keys expr_typs in
  (if List.length expressions < 1 then failwith "no expressions to optimize!");
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
    |> Optimize.insert_computations expr_typs expressions
  in
  begin match dot_after with
  | Some dot_after -> Cfg.dump_to_dot graph dot_after
  | None -> ()
  end;
  let prog = Print.decfg graph in
  Parser.print_bril Out_channel.stdout prog

let open_in_opt = function
  | Some path -> In_channel.create path
  | None -> In_channel.stdin

let command =
  let spec =
    let open Command.Spec in
    empty
    +> flag "-dot" (optional string) ~doc:"<base> Output CFG to base-before.dot and base-after.dot files"
    +> anon (maybe ("brilfile" %:string))
  in
  Command.basic_spec
    ~summary:"lcm: Lazy code motion for the Bril intermediate language"
    spec
    (fun dot_path bril_path ->
      let bril = open_in_opt bril_path in
      let dot_before, dot_after =
        match dot_path with
        | Some dot_path ->
           let dot_before = Out_channel.create @@ dot_path ^ "-before.dot" in
           let dot_after = Out_channel.create @@ dot_path ^ "-after.dot"in
           Some dot_before, Some dot_after
        | None -> None, None
      in
      go dot_before dot_after bril)

let () =
  Command.run ~version:"0.1.1" command
