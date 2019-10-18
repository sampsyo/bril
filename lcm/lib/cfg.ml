type instruction = 
  | ValueInstr of Bril.value_op
  | Print of Ident.var list
  | ConstInstr of Bril.const_op
                
type basic_block =
  { label: Ident.lbl;
    body: instruction list;
    term: Bril.term_op }

module BasicBlock = struct
  type t = basic_block
  let compare x y =
    Ident.cmp_lbl x.label y.label
  let hash x =
    Hashtbl.hash x.label
  let equal x y =
    x.label = y.label
end

include Graph.Persistent.Digraph.Concrete(BasicBlock)
