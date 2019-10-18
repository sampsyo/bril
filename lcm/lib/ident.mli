type var [@@deriving sexp]
type lbl [@@deriving sexp]
type fn [@@deriving sexp]
val var_of_string : string -> var
val lbl_of_string : string -> lbl
val fn_of_string : string -> fn
val cmp_var : var -> var -> int
val cmp_lbl : lbl -> lbl -> int
val cmp_fn : fn -> fn -> int
