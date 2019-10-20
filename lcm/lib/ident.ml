open Core

type var = string [@@deriving sexp]
type lbl = string [@@deriving sexp]
type fn = string [@@deriving sexp]
let var_of_string s = s 
let lbl_of_string s = s 
let string_of_lbl l = l
let fn_of_string s = s 
let cmp_var = String.compare
let cmp_lbl = String.compare
let cmp_fn = String.compare

let fresh_ref = ref 0

let get_fresh () =
  let v = !fresh_ref in
  fresh_ref := v + 1;
  string_of_int v

let fresh_lbl base = base ^ get_fresh ()
