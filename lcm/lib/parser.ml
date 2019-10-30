open Core

exception BrilJSONParseError of string

let parse_err msg =
  raise @@ BrilJSONParseError msg

let parse_typ : string -> Bril.typ =
  function
  | "int" -> Int
  | "bool" -> Bool
  | _ -> parse_err "not a typ"

let print_typ : Bril.typ -> Yojson.Basic.t =
  function
  | Int -> `String "int"
  | Bool -> `String "bool"

let parse_value : Yojson.Basic.t -> Bril.value =
  function
  | `Int i -> Int i
  | `Bool b -> Bool b
  | _ -> parse_err "not a value"

let print_value : Bril.value -> Yojson.Basic.t =
  function
  | Int i -> `Int i
  | Bool b -> `Bool b

let parse_var =
  function
  | `String s -> Ident.var_of_string s
  | _ -> parse_err "not an ident"

let print_var i =
  `String (Ident.string_of_var i)

let get_string =
  function
  | `String s -> s
  | _ -> parse_err "expected string"

let parse_value_expr (opcode: string) (args: Yojson.Basic.t list): Bril.value_expr =
  let args = List.map ~f:parse_var args in
  match opcode, args with
  | "add", [arg1; arg2] -> Add (arg1, arg2)
  | "mul", [arg1; arg2] -> Mul (arg1, arg2)
  | "sub", [arg1; arg2] -> Sub (arg1, arg2)
  | "div", [arg1; arg2] -> Div (arg1, arg2)
  | "eq", [arg1; arg2] -> Eq (arg1, arg2)
  | "lt", [arg1; arg2] -> Lt (arg1, arg2)
  | "gt", [arg1; arg2] -> Gt (arg1, arg2)
  | "le", [arg1; arg2] -> Le (arg1, arg2)
  | "ge", [arg1; arg2] -> Ge (arg1, arg2)
  | "not", [arg] -> Not arg
  | "and", [arg1; arg2] -> And (arg1, arg2)
  | "or", [arg1; arg2] -> Or (arg1, arg2)
  | "id", [arg] -> Id arg
  | _ -> parse_err "arity mismatch or unknown opcode for value expr"

let print_value_expr : Bril.value_expr -> Yojson.Basic.t * Yojson.Basic.t list =
  function
  | Add (arg1, arg2) -> `String "add", [print_var arg1; print_var arg2]
  | Mul (arg1, arg2) -> `String "mul", [print_var arg1; print_var arg2]
  | Sub (arg1, arg2) -> `String "sub", [print_var arg1; print_var arg2]
  | Div (arg1, arg2) -> `String "div", [print_var arg1; print_var arg2]
  | Eq (arg1, arg2) -> `String "eq", [print_var arg1; print_var arg2]
  | Lt (arg1, arg2) -> `String "lt", [print_var arg1; print_var arg2]
  | Gt (arg1, arg2) -> `String "gt", [print_var arg1; print_var arg2]
  | Le (arg1, arg2) -> `String "le", [print_var arg1; print_var arg2]
  | Ge (arg1, arg2) -> `String "ge", [print_var arg1; print_var arg2]
  | Not arg -> `String "not", [print_var arg]
  | And (arg1, arg2) -> `String "and", [print_var arg1; print_var arg2]
  | Or (arg1, arg2) -> `String "or", [print_var arg1; print_var arg2]
  | Id arg -> `String "id", [print_var arg]
  | Nop -> `String "nop", []

let parse_value_op (opcode: string) (args: Yojson.Basic.t list) (dest: string) (typ: string) : Bril.value_op =
  let value_expr = parse_value_expr opcode args in
  let dest = Ident.var_of_string dest in
  let typ = parse_typ typ in
  { op = value_expr;
    dest = dest;
    typ = typ }

let print_value_op (v : Bril.value_op) =
  let opcode, args = print_value_expr v.op in
  let dest = print_var v.dest in
  let typ = print_typ v.typ in
  `Assoc [("op", opcode);
          ("args", `List args);
          ("dest", dest);
          ("type", typ)]
  
let parse_term_op (opcode: string) (args: Yojson.Basic.t list): Bril.term_op =
  let args = List.map ~f:get_string args in
  match opcode, args with
  | "br", [cond; true_lbl; false_lbl] ->
     Br { cond = Ident.var_of_string cond;
          true_lbl = Ident.lbl_of_string true_lbl;
          false_lbl = Ident.lbl_of_string false_lbl }
  | "jmp", [lbl] ->
     Jmp (Ident.lbl_of_string lbl)
  | "ret", [] ->
     Ret
  | _ -> parse_err "arity mismatch or unknown opcode for term op (br, jmp, ret)"

let print_lbl l = 
  `String (Ident.string_of_lbl l)

let print_term_op : Bril.term_op -> Yojson.Basic.t =
  function
  | Br { cond; true_lbl; false_lbl } ->
     let cond = print_var cond in
     let t = print_lbl true_lbl in
     let f = print_lbl false_lbl in
     `Assoc [("op", `String "br");
             ("args", `List [cond; t; f])]
  | Jmp l ->
     `Assoc [("op", `String "jmp");
             ("args", `List [print_lbl l])]
  | Ret -> 
     `Assoc [("op", `String "ret");
             ("args", `List [])]

let parse_effect_op (opcode: string) (args: Yojson.Basic.t list): Bril.effect_op =
  match opcode with
  | "print" ->
     Print (List.map ~f:parse_var args)
  | _ ->
     TermOp (parse_term_op opcode args)

let print_effect_op : Bril.effect_op -> Yojson.Basic.t =
  function
  | Print args -> 
     let args = `List (List.map ~f:print_var args) in
     `Assoc [("op", `String "print");
             ("args", args)]
  | TermOp t ->
     print_term_op t

let parse_const_op (dest: string) (value: Yojson.Basic.t) (typ: string) : Bril.const_op =
  { dest = Ident.var_of_string dest;
    value = parse_value value;
    typ = parse_typ typ }

let print_const_op (c : Bril.const_op) : Yojson.Basic.t =
  `Assoc [("op", `String "const");
          ("dest", print_var c.dest);
          ("value", print_value c.value);
          ("type", print_typ c.typ)]

let parse_instruction (opcode: string) (props: (string * Yojson.Basic.t) list) : Bril.instruction =
  let args = List.Assoc.find ~equal:(=) props "args" in
  let dest = List.Assoc.find ~equal:(=) props "dest" in
  let typ = List.Assoc.find ~equal:(=) props "type" in
  let value = List.Assoc.find ~equal:(=) props "value" in
  match opcode with
  | "add"
  | "mul"
  | "sub"
  | "div"
  | "eq"
  | "lt"
  | "gt"
  | "le"
  | "ge"
  | "not"
  | "and"
  | "or"
  | "id" ->
     begin match args, dest, typ with
     | Some (`List args), Some (`String dest), Some (`String typ) ->
        ValueInstr (parse_value_op opcode args dest typ)
     | _ -> parse_err "need args, dest, and type fields for value operation"
     end

  | "nop" -> Nop

  | "br"
  | "jmp"
  | "ret"
  | "print" ->
     begin match args with
     | Some (`List args) ->
        EffectInstr (parse_effect_op opcode args)
     | _ -> parse_err "need args for effectful operation (jmp, br, ret, or print)"
     end

  | "const" ->
     begin match value, dest, typ with
     | Some value, Some (`String dest), Some (`String typ) ->
        ConstInstr (parse_const_op dest value typ)
     | _ -> parse_err "need dest, value, and type for const operation"
     end
  | op -> parse_err (Printf.sprintf "unkown opcode %s encountered" op)

let print_instruction : Bril.instruction -> Yojson.Basic.t = 
  function
  | ValueInstr v -> print_value_op v
  | EffectInstr e -> print_effect_op e
  | ConstInstr c -> print_const_op c
  | Nop -> `Assoc [("op", `String "nop")]

let parse_directive (json: Yojson.Basic.t) : Bril.directive =
  match json with
  | `Assoc l ->
     begin match List.Assoc.find ~equal:(=) l "op",
                 List.Assoc.find ~equal:(=) l "label" with
     | Some (`String opcode), _ ->
        Instruction (parse_instruction opcode l)
     | _, Some (`String label) ->
        Label (Ident.lbl_of_string label)
     | _ -> parse_err "expected op/label field to exist and be a string"
     end
  | _ -> parse_err "expected op to have keys"

let print_directive : Bril.directive -> Yojson.Basic.t =
  function
  | Instruction i -> print_instruction i
  | Label l -> `Assoc [("label", print_lbl l)]

let parse_function (json: Yojson.Basic.t) : Bril.fn =
  match Yojson.Basic.sort json with
  | `Assoc [("instrs", `List instrs); ("name", `String name)] ->
     { name = Ident.fn_of_string name;
       body = List.map ~f:parse_directive instrs }
  | _ -> parse_err "expected function json to have exactly two keys 'instrs', 'name'"

let print_fn f =
  `String (Ident.string_of_fn f)

let print_function (f: Bril.fn) : Yojson.Basic.t =
  `Assoc [("name", print_fn f.name);
          ("instrs", `List (List.map ~f:print_directive f.body))]

let parse_program (json: Yojson.Basic.t) : Bril.program =
  match json with
  | `Assoc [("functions", `List fns)] ->
     List.map ~f:parse_function fns
  | _ -> parse_err "expected top level json to have a single key 'functions'"

let print_program p : Yojson.Basic.t =
  `Assoc [("functions", `List (List.map ~f:print_function p))]

let parse_bril (file: In_channel.t): Bril.program =
  let json = Yojson.Basic.from_channel file in
  parse_program json

let print_bril (file: Out_channel.t) (bril: Bril.program) : unit =
  Yojson.Basic.to_channel file (print_program bril)
