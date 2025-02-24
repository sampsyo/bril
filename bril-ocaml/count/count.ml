open! Core
open! Bril

let () =
  let (ints, bools, floats, ptrs) =
    In_channel.input_all In_channel.stdin
    |> Yojson.Basic.from_string
    |> Bril.from_json
    |> List.fold ~init:(0, 0, 0, 0) ~f:(fun (ints, bools, floats, ptrs) func ->
      func
      |> Bril.Func.instrs
      |> List.fold ~init:(ints, bools, floats, ptrs) ~f:(fun (ints, bools, floats, ptrs) ->
          function
          | Const ((_, bril_type), _)
          | Binary ((_, bril_type), _, _, _)
          | Unary ((_, bril_type), _, _)
          | Call (Some (_, bril_type), _, _)
          | Phi ((_, bril_type), _)
          | Alloc ((_, bril_type), _)
          | Load ((_, bril_type), _)
          | PtrAdd ((_, bril_type), _, _) ->
            (match bril_type with
            | IntType -> (ints + 1, bools, floats, ptrs)
            | BoolType -> (ints, bools + 1, floats, ptrs)
            | FloatType -> (ints, bools, floats + 1, ptrs)
            | PtrType _ -> (ints, bools, floats, ptrs + 1))
          | _ -> (ints, bools, floats, ptrs)))
  in
  printf "Ints: %d Bools: %d Floats: %d Pointers: %d \n" ints bools floats ptrs
