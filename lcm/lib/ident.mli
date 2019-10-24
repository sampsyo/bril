type var [@@deriving sexp, compare]
type lbl [@@deriving sexp, compare]
type fn [@@deriving sexp, compare]
val var_of_string : string -> var
val string_of_var : var -> string
val lbl_of_string : string -> lbl
val string_of_lbl : lbl -> string
val fn_of_string : string -> fn
val string_of_fn : fn -> string
val cmp_var : var -> var -> int
val cmp_lbl : lbl -> lbl -> int
val cmp_fn : fn -> fn -> int
val fresh_lbl : string -> lbl
