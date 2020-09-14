open Core

let () =
  let input = In_channel.input_all In_channel.stdin in
  let json = Yojson.Basic.from_string input in
  let open Yojson.Basic.Util in
  let (ints, bools) =
    json |> member "functions" |> to_list
    |> List.fold ~init:(0, 0) ~f:(fun (ints, bools) json ->
        json |> member "instrs" |> to_list
        |> List.fold ~init:(ints, bools) ~f:(fun (ints, bools) json ->
            match json |> member "type" |> to_string_option with
            | Some "int" -> (ints + 1, bools)
            | Some "bool" -> (ints, bools + 1)
            | Some _
            | None -> (ints, bools)))
  in
  printf "Ints: %d Bools: %d\n" ints bools
