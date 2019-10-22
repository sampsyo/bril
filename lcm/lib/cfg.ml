open Core

type instruction = 
  | ValueInstr of Bril.value_op
  | Print of Ident.var list
  | ConstInstr of Bril.const_op
  | Nop
  [@@deriving sexp]
                
type pre_basic_block =
  { pre_lbl: Ident.lbl option;
    pre_body: instruction list;
    pre_term: Bril.term_op option }
  [@@deriving sexp]

type basic_block =
  { lbl: Ident.lbl;
    body: instruction list;
    term: Bril.term_op }
  [@@deriving sexp]

module Attrs = struct
  type t = (string, Bitv.t) Hashtbl.t
  let create () = String.Table.create ()
  let get = Hashtbl.find_exn
end

module BasicBlock = struct
  type t = basic_block * Attrs.t
  let compare (x, _) (y, _) =
    Ident.cmp_lbl x.lbl y.lbl
  let hash (x, _) =
    Hashtbl.hash x.lbl
  let equal (x, _) (y, _) =
    x.lbl = y.lbl
end

module EdgeLabel = struct
  type t = Attrs.t
  let compare x y =
    compare (Hashtbl.length x) (Hashtbl.length y)
  let default : t = String.Table.create ()
end

module Label = struct
  type t = Ident.lbl [@@deriving sexp]
  let compare = Ident.cmp_lbl
  let hash = Hashtbl.hash
  let equal = (=)
end

module CFG = struct
  include Graph.Persistent.Digraph.ConcreteLabeled(BasicBlock)(EdgeLabel)
  let vertex_name (v, _) = Ident.string_of_lbl v.lbl
  let graph_attributes _ = []
  let default_vertex_attributes _ = []

  let vertex_attributes (block, _) : Graph.Graphviz.DotAttributes.vertex list =
    let text =
      let label = Ident.string_of_lbl block.lbl in
      let body = Sexp.to_string ([%sexp_of: instruction list] block.body) in
      let term = Sexp.to_string ([%sexp_of: Bril.term_op] block.term) in
      label ^ "\n" ^ body ^ "\n" ^ term
    in
    
    [`Shape `Box;
     `Label text]

  let default_edge_attributes _ = []
  let edge_attributes _ = []
  let get_subgraph _ = None
end

module Map = Map.Make(Label)
type t = CFG.t * BasicBlock.t Map.t

let empty : t = (CFG.empty, Map.empty)

let add_block (g, m) block : t =
  let block' = (block, Attrs.create ()) in
  let g = CFG.add_vertex g block' in
  let m = Map.add_exn ~key:block.lbl ~data:block' m in
  (g, m)

module Viz = Graph.Graphviz.Dot(CFG)

let dump_to_dot (graph: CFG.t) (channel: Out_channel.t) =
  Viz.output_graph channel graph
