open! Core

module Make (M : sig
  type t [@@deriving compare, equal, sexp_of]

  val by_name : (string * t) list
end) =
struct
  include M

  let is_op = List.Assoc.mem by_name ~equal:String.equal
  let of_string = List.Assoc.find_exn by_name ~equal:String.equal
  let to_string = List.map by_name ~f:(fun (a, b) -> (b, a)) |> List.Assoc.find_exn ~equal
end

module Binary = struct
  module T = struct
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
    [@@deriving compare, equal, sexp_of]

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
  end

  include Make (T)
  include T
end

module Unary = struct
  module T = struct
    type t =
      | Not
      | Id
    [@@deriving compare, equal, sexp_of]

    let by_name = [ ("not", Not); ("id", Id) ]
  end

  include Make (T)
  include T
end
