open Core

type var = string [@@deriving sexp, compare]
type lbl = string [@@deriving sexp, compare]
type fn = string [@@deriving sexp, compare]
let var_of_string s = s 
let string_of_var s = s 
let lbl_of_string s = s 
let string_of_lbl s = s
let fn_of_string s = s 
let string_of_fn s = s 
let cmp_var = String.compare
let cmp_lbl = String.compare
let cmp_fn = String.compare

let fresh_ref = ref 0

let get_fresh () =
  let v = !fresh_ref in
  fresh_ref := v + 1;
  string_of_int v

let fresh_lbl base = base ^ get_fresh ()