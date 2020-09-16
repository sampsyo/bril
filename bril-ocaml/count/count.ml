open! Core
open! Bril

let () =
  let input = In_channel.input_all In_channel.stdin in
  let { funcs } = from_string input in
  let (ints, bools) =
    List.fold funcs ~init:(0, 0) ~f:(fun (ints, bools) { instrs; _ } ->
        List.fold instrs ~init:(ints, bools) ~f:(fun (ints, bools) -> function
          | Const ((_, bril_type), _)
          | Binary ((_, bril_type), _, _, _)
          | Unary ((_, bril_type), _, _)
          | Call (Some (_, bril_type), _, _) -> (
              match bril_type with
              | IntType -> (ints + 1, bools)
              | BoolType -> (ints, bools + 1) )
          | _ -> (ints, bools)))
  in
  printf "Ints: %d Bools: %d\n" ints bools
