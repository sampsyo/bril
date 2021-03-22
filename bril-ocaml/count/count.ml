open! Core
open! Bril

let () =
  let (ints, bools, ptrs) =
    In_channel.input_all In_channel.stdin
    |> Yojson.Basic.from_string
    |> Bril.from_json
    |> List.fold ~init:(0, 0, 0) ~f:(fun (ints, bools, ptrs) func ->
           func
           |> Bril.Func.instrs
           |> List.fold ~init:(ints, bools, ptrs) ~f:(fun (ints, bools, ptrs) -> function
                | Const ((_, bril_type), _)
                | Binary ((_, bril_type), _, _, _)
                | Unary ((_, bril_type), _, _)
                | Call (Some (_, bril_type), _, _) ->
                  ( match bril_type with
                  | IntType -> (ints + 1, bools, ptrs)
                  | BoolType -> (ints, bools + 1, ptrs)
                  | PtrType _ -> (ints, bools, ptrs + 1))
                | _ -> (ints, bools, ptrs)))
  in
  printf "Ints: %d Bools: %d Pointers: %d \n" ints bools ptrs
