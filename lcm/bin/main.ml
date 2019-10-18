open Core
open Lcm

let fu _ =
  failwith "unimplemented"


let build_cfg (prog: Bril.program) : Cfg.t =
  fu prog

let analyze (cfg: Cfg.t) =
  fu cfg

let go file () =
  print_s ([%sexp_of: Bril.program] (Parser.parse_bril file))

let command =
  let spec =
    let open Command.Spec in
    empty +> anon (maybe ("brilfile" %:string))
  in
  Command.basic_spec
    ~summary:"lcm: Lazy code motion for the Bril intermediate language"
    spec
    (function
     | Some path -> go @@ In_channel.create path
     | None -> go In_channel.stdin)

let () =
  Command.run ~version:"0.1.1" command
