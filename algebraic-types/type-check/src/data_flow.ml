
(* modified from my lesson 4 submission. *)

type direction = Forward | Backward

type 'a cfg_node = {
  instr : Bril.Instr.t;
  label : string;
  mutable incoming : 'a cfg_node list;
  mutable outgoing : 'a cfg_node list;
  mutable workset : 'a;
  mutable prev_workset : 'a;
}

module type FlowAnalysis = sig
  type workset
  type extra

  val direction : direction

  val meet : extra -> workset -> workset -> workset

  val transfer : extra -> workset -> workset cfg_node -> workset

  val compare : workset -> workset -> int

  val print : workset -> string
end

module CfgConstructor (F : FlowAnalysis) = struct
  let construct_cfg
      (top : F.workset)
      (func : Bril.Func.t)
    : F.workset cfg_node list =

    let open F in
    let cfg_node_map, cfg_nodes =
      List.fold_left
        begin fun (acc, nodes) block_name ->
          let instrs = Core.String.Map.find_exn func.blocks block_name in
          let cfg_nodes =
            List.map (fun instr -> {
                  instr;
                  label = block_name;
                  incoming = [];
                  outgoing = [];
                  workset = top;
                  prev_workset = top;
                }) instrs in
          let rec helper = function
            | hd :: md :: tl ->
              hd.outgoing <- md :: hd.outgoing;
              md.incoming <- hd :: md.incoming;
              helper (md :: tl)
            | _ -> () in
          helper cfg_nodes;
          if List.length cfg_nodes > 0 then begin
            Core.String.Map.add_exn acc block_name
              (cfg_nodes |> List.hd, List.rev cfg_nodes |> List.hd),
            nodes @ cfg_nodes
          end else acc, nodes @ cfg_nodes
        end (Core.String.Map.empty, []) func.order in

    List.iter2
      begin fun (first, last) block_name ->
        let succ_nodes =
          Core.String.Map.find func.succs block_name
          |> Option.value ~default:[]
          |> (List.filter_map (Core.String.Map.find cfg_node_map)) in
        last.outgoing <- (List.map fst succ_nodes) @ last.outgoing;
        List.iter (fun succ_node ->
            succ_node.incoming <- last :: succ_node.incoming)
          (List.map fst succ_nodes)
      end (Core.String.Map.data cfg_node_map) func.order;
    cfg_nodes
end

module DataFlow (F : FlowAnalysis) = struct
  let perform_analysis (extra : F.extra) (top : F.workset) (cfg : F.workset cfg_node list) : unit =
    let open F in
    let worklist = Queue.create () in
    List.iter (fun cfg_node -> Queue.push cfg_node worklist) cfg;
    match cfg with
    | [] -> () (* special-case empty cfgs *)
    | _ ->
      while not (Queue.is_empty worklist) do
        let cfg_node = Queue.pop worklist in
        let in_set =
          begin
            match direction with
            | Forward -> cfg_node.incoming
            | Backward -> cfg_node.outgoing
          end
          |> List.map (fun cfg_node -> cfg_node.workset)
          |> List.fold_left (meet extra) top in
        cfg_node.prev_workset <- in_set;
        (* transfer function *)
        let workset' = transfer extra in_set cfg_node in
        if compare workset' cfg_node.workset <> 0 then begin
          cfg_node.workset <- workset';
          (* add successors to queue *)
          match F.direction with
          | Forward ->
            List.iter (fun node -> Queue.push node worklist) cfg_node.outgoing
          | Backward ->
            List.iter (fun node -> Queue.push node worklist) cfg_node.incoming
        end
      done
end
