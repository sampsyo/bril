open Core

type var = string [@@deriving sexp]
type lbl = string [@@deriving sexp]
type fn = string [@@deriving sexp]
let var_of_string s = s 
let lbl_of_string s = s 
let fn_of_string s = s 
let cmp_var = String.compare
let cmp_lbl = String.compare
let cmp_fn = String.compare
