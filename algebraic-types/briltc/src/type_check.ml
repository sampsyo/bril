
open! Core
open Bril
open Data_flow

module TypeCheckAnalysis = struct
  type workset = Bril.Bril_type.t String.Map.t * (string * Bril.Bril_type.t String.Map.t) option

  type extra = {
    func : Bril.Func.t;
    func_map : Bril.Func.t String.Map.t;
  }

  let direction = Forward

  let meet _ (workset, helper) (workset', helper') =
    let workset'' =
      String.Map.merge
        workset workset'
        ~f:begin
          fun ~key -> function
            | `Both (left, right) ->
              if Bril_type.equal left right
              then Some left
              else begin
                prerr_endline (sprintf "multiple types for var %s, %s and %s" key
                                 (Bril_type.to_string left) (Bril_type.to_string right));
                None
              end
            | `Left typ | `Right typ -> Some typ
        end in
    let helper'' =
      match helper, helper' with
      | None, None -> None
      | Some h, None | None, Some h -> Some h
      | Some _, Some _ -> failwith "label cannot be the target of multiple destructs" in
    workset'', helper''

  let transfer {func; func_map} ((type_map, destruct_helper) as workset) (cfg_node : workset cfg_node) =
    let open Bril.Instr in
    let type_map, destruct_helper =
      match destruct_helper with
      | Some (var, map) ->
        begin
          match String.Map.find map cfg_node.label with
          | Some typ ->
            let type_map' = String.Map.update type_map var ~f:(fun _ -> typ) in
            type_map', None
          | None -> failwithf "label not an option from destruct: %s" cfg_node.label ()
        end
      | None -> type_map, None in
    let update_type (var, typ) =
      String.Map.update type_map var ~f:(fun _ -> typ), destruct_helper in
    let lookup_type var =
      match String.Map.find type_map var with
      | Some typ -> typ
      | None -> failwithf "undefined var: %s" var () in
    match cfg_node.instr with
    | Label _ -> workset

    | Const ((_, IntType) as dest, Int _)
    | Const ((_, BoolType) as dest, Bool _) ->
      update_type dest

    | Const _ -> failwith "bad const typing"

    | Binary ((_, IntType) as dest, (Add | Mul | Sub | Div), lhs, rhs) ->
      if Bril_type.equal (lookup_type lhs) IntType &&
         Bril_type.equal (lookup_type rhs) IntType
      then update_type dest
      else failwith "must have ints on right side of arithmetic instruction"

    | Binary ((_, BoolType) as dest, (Eq | Lt | Gt | Le | Ge), lhs, rhs) ->
      if Bril_type.equal (lookup_type lhs) IntType &&
         Bril_type.equal (lookup_type rhs) IntType
      then update_type dest
      else failwith "must have ints on right side of comparison instruction"

    | Binary ((_, BoolType) as dest, (And | Or), lhs, rhs) ->
      if Bril_type.equal (lookup_type lhs) BoolType &&
         Bril_type.equal (lookup_type rhs) BoolType
      then update_type dest
      else failwith "must have bools on right side of boolean instruction"

    | Binary _ -> failwith "bad binary op typing"

    | Unary ((_, BoolType) as dest, Not, arg) ->
      if Bril_type.equal (lookup_type arg) BoolType
      then update_type dest
      else failwith "must have bool on right side of not instruction"

    | Unary ((_, typ) as dest, Id, arg) ->
      if Bril_type.equal (lookup_type arg) typ
      then update_type dest
      else failwith "id instruction must have same type on both sides"

    | Unary _ -> failwith "bad unary op typing"

    | Jmp _ -> workset

    | Br (arg, _, _) ->
      if Bril_type.equal (lookup_type arg) BoolType
      then workset
      else failwith "cannot branch on non-bool var"

    | Call (dest_opt, func_name, args) ->
      begin
        let called_func = String.Map.find_exn func_map func_name in

        let () =
          match dest_opt with
          | Some (_, typ) ->
            begin
              match called_func.ret_type with
              | Some ret_type ->
                if not (Bril_type.equal typ ret_type)
                then failwith "attempt to store value from a function into wrong typed variable"
              | None -> failwith "attempt to store value from function without return value"
            end
          | None -> () in

        let () =
          match List.iter2
                  args called_func.args
                  ~f:begin
                    fun arg (_, typ) ->
                      if not (Bril_type.equal (lookup_type arg) typ)
                      then failwith "invalid argument type for function call"
                  end with
          | Ok () -> ()
          | Unequal_lengths -> failwith "attempt to call function with incorrect number of arguments" in

        Option.map ~f:update_type dest_opt |> Option.value ~default:workset
      end

    | Ret None -> workset

    | Ret (Some arg) ->
      begin
        match func.ret_type with
        | Some ret_type ->
          if Bril_type.equal (lookup_type arg) ret_type
          then workset
          else failwith "invalid return type"
        | None -> failwith "return value in function without return type"
      end

    | Print _ -> workset

    | Nop -> workset

    | Phi _ -> failwith "SSA-form unsupported"

    | Speculate | Commit | Guard _ -> failwith "speculation unsupported"

    | Pack ((_, ProdType typs) as dest, args) ->
      begin
        match List.iter2 typs args
                ~f:begin fun typ arg ->
                  if not (Bril_type.equal (lookup_type arg) typ)
                  then failwith "incompatible types in pack"
                end with
        | Ok () -> update_type dest
        | Unequal_lengths -> failwith "unequal lengths of product type and args for pack"
      end

    | Pack _ -> failwith "cannot pack into a non-product type"

    | Unpack ((_, typ) as dest, arg, index) ->
      begin
        match lookup_type arg with
        | ProdType typs ->
          begin
            match List.nth typs index with
            | Some typ' ->
              if Bril_type.equal typ typ'
              then update_type dest
              else failwith "incomaptible types in unpack"
            | None -> failwith "unpack index out of bounds"
          end
        | _ -> failwith "cannot unpack a non-product type"
      end

    | Construct ((_, SumType typs) as dest, arg, index) ->
      let typ = lookup_type arg in
      if Bril_type.equal typ (List.nth_exn typs index)
      then update_type dest
      else failwith "invalid type; does not match type at specified \
                     index in product type"

    | Construct _ -> failwith "cannot construct a non-sum type"

    | Destruct ((var, SumType typs), arg, labels) ->
      begin
        if not (Bril_type.equal (String.Map.find_exn type_map arg) (SumType typs))
        then failwith "mismatched destruct type";
        match List.fold2
                labels typs
                ~init:String.Map.empty
                ~f:(fun acc label typ -> String.Map.add_exn acc ~key:label ~data:typ) with
        | Ok destruct_helper' -> type_map, Some (var, destruct_helper')
        | Unequal_lengths -> failwith "unequal lengths of sum type and labels for destruct"
      end

    | Destruct _ -> failwith "destruct must be annotated with a sum type"

  let print (workset, _) =
    String.Map.fold workset ~init:[]
      ~f:(fun ~key ~data acc -> (key ^ ": " ^ Bril_type.to_string data) :: acc)
    |> List.rev |> String.concat ~sep:"; "

  let compare (w, _) (w', _) =
    String.Map.compare Bril_type.compare w w'
end

module TypeCheckCfgConstructor = CfgConstructor (TypeCheckAnalysis)
module TypeCheckDataFlow = DataFlow (TypeCheckAnalysis)

let perform (func_map : Bril.Func.t String.Map.t) func =
  let top_workset = List.fold Bril.Func.(func.args) ~init:String.Map.empty
      ~f:(fun acc (var, typ) -> String.Map.add_exn acc ~key:var ~data:typ) in
  let top = top_workset, None in
  let cfg = TypeCheckCfgConstructor.construct_cfg top func in
  let extra = TypeCheckAnalysis.{func; func_map} in
  TypeCheckDataFlow.perform_analysis extra top cfg
