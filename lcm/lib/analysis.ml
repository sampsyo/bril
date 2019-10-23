open Core
open Cfg

module type Analysis = sig
  val run : CFG.t -> CFG.t
end

module type DataflowAnalysis = sig
  val attr_name : string
  val direction : Graph.Fixpoint.direction
  val analyze : CFG.E.t -> Bitv.t -> Bitv.t
  val init : CFG.V.t -> Bitv.t
end

module type BlockLocalAnalysis = sig
  val attr_name : string
  val analyze : CFG.V.t -> Bitv.t
end

module type EdgeLocalAnalysis = sig
  val attr_name : string
  val analyze : CFG.E.t -> Bitv.t
end

module MakeBlockLocal (A: BlockLocalAnalysis) = struct
  let run graph =
    let do_block b =
      let res = A.analyze b in
      Hashtbl.set ~key:A.attr_name ~data:res (snd b)
    in
    Cfg.CFG.iter_vertex do_block graph;
    graph
end

module MakeEdgeLocal (A: EdgeLocalAnalysis) = struct
  let run graph =
    let do_edge (src, attrs, dst) =
      let res = A.analyze (src, attrs, dst) in
      Hashtbl.set ~key:A.attr_name ~data:res attrs
    in
    Cfg.CFG.iter_edges_e do_edge graph;
    graph
end

module MakeDataflow (A: DataflowAnalysis) : Analysis = struct
  module AA = struct
    type data = Bitv.t
    type edge = CFG.E.t
    type vertex = CFG.V.t
    type g = CFG.t
    let join = Bitv.bw_and
    let equal x y = Bitv.all_zeros (Bitv.bw_xor x y)
    include A
  end
  include Graph.Fixpoint.Make(CFG)(AA)
  let run graph =
    let data = analyze A.init graph in
    let update_block b =
      Hashtbl.set ~key:A.attr_name ~data:(data b) (snd b)
    in
    Cfg.CFG.iter_vertex update_block graph;
    graph
end

module BitvAnalysis = struct
  type data = Bitv.t
  type edge = CFG.E.t
  type vertex = CFG.V.t
  type g = CFG.t
  let join = Bitv.bw_and
  let equal x y = Bitv.all_zeros (Bitv.bw_xor x y)
end
