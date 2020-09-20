open! Core

let make_by_op = List.map ~f:(fun (a, b) -> (b, a))

module Binary = struct
  type t =
    | Add
    | Mul
    | Sub
    | Div
    | Eq
    | Lt
    | Gt
    | Le
    | Ge
    | And
    | Or
  [@@deriving compare, sexp_of, equal]

  let by_name =
    [
      ("add", Add);
      ("mul", Mul);
      ("sub", Sub);
      ("div", Div);
      ("eq", Eq);
      ("lt", Lt);
      ("gt", Gt);
      ("le", Le);
      ("ge", Ge);
      ("and", And);
      ("or", Or);
    ]

  let by_op = make_by_op by_name
end

module Unary = struct
  type t =
    | Not
    | Id
  [@@deriving compare, sexp_of, equal]

  let by_name = [ ("not", Not); ("id", Id) ]
  let by_op = make_by_op by_name
end
