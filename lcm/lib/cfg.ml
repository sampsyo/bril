type instruction = 
  | ValueInstr of Bril.value_op
  | Print of Ident.var list
  | ConstInstr of Bril.const_op
  | Nop
                
type pre_basic_block =
  { pre_lbl: Ident.lbl option;
    pre_body: instruction list;
    pre_term: Bril.term_op option }

type basic_block =
  { lbl: Ident.lbl;
    body: instruction list;
    term: Bril.term_op }

module BasicBlock = struct
  type t = basic_block
  let compare x y =
    Ident.cmp_lbl x.lbl y.lbl
  let hash x =
    Hashtbl.hash x.lbl
  let equal x y =
    x.lbl = y.lbl
end

module Label = struct
  type t = Ident.lbl
  let compare = Ident.cmp_lbl
  let hash = Hashtbl.hash
  let equal = (=)
end

module Graph = Graph.Persistent.Digraph.Concrete(BasicBlock)
module Map = Map.Make(Label)
type t = Graph.t * basic_block Map.t

let empty : t = (Graph.empty, Map.empty)
let add_block (g, m) block =
  let g = Graph.add_vertex g block in
  let m = Map.add block.lbl block m in
  (g, m)
