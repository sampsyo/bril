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

  let fold_int t n1 n2 =
    match t with
    | Add -> Const.Int (n1 + n2)
    | Mul -> Const.Int (n1 * n2)
    | Sub -> Const.Int (n1 - n2)
    | Div -> Const.Int (n1 / n2)
    | Eq -> Const.Bool (n1 = n2)
    | Lt -> Const.Bool (n1 < n2)
    | Gt -> Const.Bool (n1 > n2)
    | Le -> Const.Bool (n1 <= n2)
    | Ge -> Const.Bool (n1 >= n2)
    | _ -> failwithf "[Op.Binary.fold]: type mismatch for operation %s %d %d" (to_string t) n1 n2 ()

  let fold_bool t b1 b2 =
    match t with
    | And -> Const.Bool (b1 && b2)
    | Or -> Const.Bool (b1 || b2)
    | _ -> failwithf "[Op.Binary.fold]: type mismatch for operation %s %b %b" (to_string t) b1 b2 ()

  let fold t v1 v2 =
    match (v1, v2) with
    | (Const.Int n1, Const.Int n2) -> fold_int t n1 n2
    | (Const.Bool b1, Const.Bool b2) -> fold_bool t b1 b2
    | _ ->
      failwithf
        "[Op.Binary.fold]: type mismatch between arguments %s and %s"
        (Const.to_string v1)
        (Const.to_string v2)
        ()
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

  let fold t v =
    match (t, v) with
    | (Not, Const.Bool b) -> Const.Bool (not b)
    | (Id, _) -> v
    | _ ->
      failwithf
        "[Op.Unary.fold]: type mismatch for operation %s %s"
        (to_string t)
        (Const.to_string v)
        ()
end
