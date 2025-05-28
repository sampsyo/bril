#![allow(unused_variables)]
use core::panic;
use std::collections::HashMap;
use std::str;

use crate::types::*;

// An environment maps variable names (`&str`s) to values
pub type Environment<'a> = HashMap<&'a str, BrilValue>;

/// Extracts the variable name (string) that occupies `start_idx` to `end_idx`
/// (inclusive) in `instr_view.var_store`
pub fn get_var<'a>(
    instr_view: &'a InstrView,
    start_idx: u32,
    end_idx: u32,
) -> &'a str {
    let start_idx = start_idx as usize;
    let end_idx = end_idx as usize;
    str::from_utf8(&instr_view.var_store[start_idx..=end_idx])
        .expect("invalid utf-8")
}

/// Extracts a vec of args (variable name strings) that correspond to the
/// `args_start` to `args_end` indices (inclusive) in `instr_view.arg_idxes_store`
pub fn get_args<'a>(
    instr_view: &'a InstrView,
    args_start: u32,
    args_end: u32,
) -> Vec<&'a str> {
    let arg_start = args_start as usize;
    let arg_end = args_end as usize;
    let args_idxes_slice = &instr_view.arg_idxes_store[arg_start..=arg_end];
    args_idxes_slice
        .iter()
        .map(|i32pair| {
            let (start_idx, end_idx) = <(u32, u32)>::from(*i32pair);
            get_var(instr_view, start_idx, end_idx)
        })
        .collect()
}

/// Extracts the label name (string) that occupies `start_idx` to `end_idx`
/// (inclusive) in `instr_view.labels_store`
pub fn get_label_name<'a>(
    instr_view: &'a InstrView,
    start_idx: u32,
    end_idx: u32,
) -> &'a str {
    let start_idx = start_idx as usize;
    let end_idx = end_idx as usize;
    str::from_utf8(&instr_view.labels_store[start_idx..=end_idx])
        .expect("invalid utf-8")
}

/// Extracts a vec of labels that correspond to the
/// `labels_start` to `labels_end` indices (inclusive) in `instr_view.labels_idxes_store`
pub fn get_labels_vec<'a>(
    instr_view: &'a InstrView,
    labels_start: u32,
    labels_end: u32,
) -> Vec<&'a str> {
    let label_start = labels_start as usize;
    let label_end = labels_end as usize;
    let labels_idxes_slice =
        &instr_view.labels_idxes_store[label_start..=label_end];
    labels_idxes_slice
        .iter()
        .map(|i32pair| {
            let (start_idx, end_idx) = <(u32, u32)>::from(*i32pair);
            get_label_name(instr_view, start_idx, end_idx)
        })
        .collect()
}

/// Extracts the function name (string) that occupies `start_idx` to `end_idx`
/// (inclusive) in `instr_view.funcs_store`
pub fn get_func<'a>(
    instr_view: &'a InstrView,
    start_idx: u32,
    end_idx: u32,
) -> &'a str {
    let start_idx = start_idx as usize;
    let end_idx = end_idx as usize;
    str::from_utf8(&instr_view.funcs_store[start_idx..=end_idx])
        .expect("invalid utf-8")
}

/// Returns the PC (index in the list of instrs) corresponding to a label
/// as an `Option`. (Returns `None` if no such PC exists.)
pub fn get_pc_of_label(
    instr_view: &InstrView,
    label_str: &str,
) -> Option<usize> {
    // Iterate over the list of instrs to find the index (PC)
    // of the instr corresponding to the label (we do this
    // by comparing the actual label strings)
    instr_view.instrs.iter().position(|instr| {
        if instr.op == u32::MAX {
            let candidate_label_str = get_label_name(
                instr_view,
                instr.label.first as u32,
                instr.label.second as u32,
            );
            candidate_label_str == label_str
        } else {
            false
        }
    })
}

/// Interprets a unary value operation (`not` and `id`)
/// (panics if `op` is not an unop)
pub fn interp_unop<'a>(
    instr_view: &'a InstrView,
    op: Opcode,
    instr: &FlatInstr,
    env: &mut Environment<'a>,
) {
    if !op.is_unop() {
        panic!("interp_unop called on a non-unary value operation");
    }

    let (dest_start, dest_end): (u32, u32) = instr.dest.into();
    let dest = get_var(instr_view, dest_start, dest_end);
    let (args_start, args_end): (u32, u32) = instr.args.into();
    let args = get_args(instr_view, args_start, args_end);
    assert!(
        args.len() == 1,
        "unary instruction is malformed (no. of args != 1)"
    );

    let arg = args[0];
    let value = env.get(arg).expect("arg missing from env");
    let result = match (op, value) {
        (Opcode::Not, BrilValue::BoolVal(b)) => {
            let b = bool::from(*b);
            BrilValue::BoolVal((!b).into())
        }
        (Opcode::Id, _) => *value,
        _ => {
            panic!("argument to unary instruction is ill-typed");
        }
    };
    env.insert(dest, result);
}

/// Interprets a binary value operation (panics if `op` is not a binop)
pub fn interp_binop<'a>(
    instr_view: &'a InstrView,
    op: Opcode,
    instr: &FlatInstr,
    env: &mut Environment<'a>,
) {
    use BrilValue::*;
    use Opcode::*;

    if !op.is_binop() {
        panic!("interp_binop called on a non-binary value operation");
    }

    let (dest_start, dest_end): (u32, u32) = instr.dest.into();
    let dest = get_var(instr_view, dest_start, dest_end);

    let (args_start, args_end): (u32, u32) = instr.args.into();
    let args = get_args(instr_view, args_start, args_end);
    assert!(args.len() == 2, "no. of args to arithmetic op != 2");

    let x = env.get(args[0]).expect("left operand missing from env");
    let y = env.get(args[1]).expect("right operand missing from env");

    match (x, y) {
        (IntVal(v1), IntVal(v2)) => {
            let value = match op {
                // Arithmetic
                Add => IntVal(v1.wrapping_add(*v2)),
                Sub => IntVal(v1.wrapping_sub(*v2)),
                Mul => IntVal(v1.wrapping_mul(*v2)),
                Div => IntVal(v1.wrapping_div(*v2)),
                // Comparison
                Eq => BoolVal((v1 == v2).into()),
                Ge => BoolVal((v1 >= v2).into()),
                Gt => BoolVal((v1 > v2).into()),
                Le => BoolVal((v1 <= v2).into()),
                Lt => BoolVal((v1 < v2).into()),
                _ => unreachable!(),
            };

            env.insert(dest, value);
        }
        (BoolVal(b1), BoolVal(b2)) => {
            let b1 = bool::from(*b1);
            let b2 = bool::from(*b2);
            // Logic
            let value = match op {
                And => BoolVal((b1 && b2).into()),
                Or => BoolVal((b1 || b2).into()),
                _ => unreachable!(),
            };
            env.insert(dest, value);
        }
        (_, _) => {
            panic!("operands to binop are ill-typed")
        }
    }
}

/// Interprets a function call
pub fn interp_call<'a>(
    instr_view: &'a InstrView,
    env: &mut Environment<'a>,
    funcs: &HashMap<&str, &InstrView>,
    instr: &FlatInstr,
    instr_kind: InstrKind,
) {
    let (funcs_start, funcs_end): (u32, u32) = instr.funcs.into();
    let func_name = get_func(instr_view, funcs_start, funcs_end);

    let call_view = funcs
        .get(func_name)
        .expect("func_name missing from funcs hashmap");

    let mut fresh_env = Environment::new();
    // No args supplied to function call, just interpret the callee
    if instr.args.first == -1 && instr.args.second == -1 {
        let possible_return_value =
            interp_instr_view(call_view, &mut fresh_env, funcs)
                .expect("error encountered when interpreting instr_view");
        match instr_kind {
            InstrKind::ValueOp => {
                // Call function
                let ret_value = possible_return_value
                    .expect("missing return value for Call ValueOp");
                let (dest_start, dest_end): (u32, u32) = instr.dest.into();
                let dest_var = get_var(instr_view, dest_start, dest_end);
                env.insert(dest_var, ret_value);
            }
            InstrKind::EffectOp => {
                // There's no dest if it's an effect-op, so we're done
            }
            _ => unreachable!(),
        }
    } else {
        // The function call has args supplied to it
        let instr_args = instr.args;

        let (args_start, args_end): (u32, u32) = instr.args.into();

        // Args supplied to call instruction
        let args = get_args(instr_view, args_start, args_end);

        let args_values: Vec<&BrilValue> = args
            .into_iter()
            .map(|a| env.get(a).expect("arg missing from env"))
            .collect();

        for (flat_arg, arg_value) in call_view.func_args.iter().zip(args_values)
        {
            // Check typing
            let (start_idx, end_idx): (u32, u32) =
                flat_arg.arg_name_idxes.into();

            // Function args
            let arg_name = get_var(call_view, start_idx, end_idx);

            let desired_arg_type: FlatType = flat_arg.arg_type;
            let actual_arg_type: FlatType = arg_value.get_type().into();
            match (desired_arg_type, actual_arg_type) {
                (FlatType::Int, FlatType::Int)
                | (FlatType::Bool, FlatType::Bool) => {
                    // Function arg is well-typed, extend the env with the arg_value
                    fresh_env.insert(arg_name, *arg_value);
                }
                (FlatType::Null, _) | (_, FlatType::Null) => {
                    panic!("encountered null type for function argument");
                }
                (_, _) => {
                    panic!(
                        "Type of supplied argument doesn't match expected type of function argument"
                    );
                }
            }
        }

        // Call function
        match instr_kind {
            InstrKind::ValueOp => {
                // Call function
                let ret_value =
                    interp_instr_view(call_view, &mut fresh_env, funcs)
                        .expect("error interpreting function call");
                let (dest_start, dest_end): (u32, u32) = instr.dest.into();
                let dest_var = get_var(instr_view, dest_start, dest_end);
                env.insert(dest_var, ret_value.expect("missing return value"));
            }
            InstrKind::EffectOp => {
                // There is no return value, so we can just ignore the result
                // of `inerp_instr_view`
                interp_instr_view(call_view, &mut fresh_env, funcs)
                    .expect("error interpreting function call");

                // There's no dest if it's an effect-op, so we're done
            }
            _ => unreachable!(),
        }
    }
}

/// Interprets all the instructions in `instr_view` using the supplied `env`
pub fn interp_instr_view<'a>(
    instr_view: &'a InstrView,
    env: &mut Environment<'a>,
    funcs: &HashMap<&str, &InstrView>,
) -> Result<Option<BrilValue>, String> {
    let func_name = str::from_utf8(instr_view.func_name).unwrap();

    let mut current_instr_ptr = 0; // Initialize program counter

    while current_instr_ptr < instr_view.instrs.len() {
        let instr = &instr_view.instrs[current_instr_ptr];
        let instr_kind = instr.get_instr_kind();
        if let InstrKind::Label = instr_kind {
            // Reached a label annotation in the program, proceed to the next line
            current_instr_ptr += 1;
            continue;
        }
        let op: Opcode = Opcode::u32_to_opcode(instr.op)
            .expect("unable to convert u32 to opcode");
        match instr_kind {
            InstrKind::Label => {
                // handled above already
                unreachable!()
            }
            InstrKind::Const => {
                let (dest_start, dest_end): (u32, u32) = instr.dest.into();
                let dest = get_var(instr_view, dest_start, dest_end);
                let value =
                    instr.value.try_into().expect("Encountered a null value");

                // Extend the environment so that `dest |-> value`
                env.insert(dest, value);
                current_instr_ptr += 1;
                continue;
            }
            InstrKind::EffectOp => {
                if let Opcode::Print = op {
                    let (args_start, args_end): (u32, u32) = instr.args.into();
                    let args = get_args(instr_view, args_start, args_end);

                    let arg_values: Vec<&BrilValue> = args
                        .iter()
                        .map(|arg| env.get(arg).expect("arg missing from env"))
                        .collect();

                    let value_strs: Vec<String> = arg_values
                        .iter()
                        .map(|value| format!("{value}"))
                        .collect();

                    let string_to_print = value_strs.join(" ");

                    // Actually print out the value of the arguments
                    // NOTE TO SELF: DO NOT REMOVE THIS PRINTLN
                    println!("{string_to_print}");

                    current_instr_ptr += 1;
                } else if let Opcode::Jmp = op {
                    // Fetch the start/end idx of the label in the `labels_store`
                    let (label_start, label_end): (u32, u32) =
                        instr.instr_labels.into();

                    // Grab the actual vector of label strings corresponding
                    // to these indices
                    let labels_vec =
                        get_labels_vec(instr_view, label_start, label_end);

                    assert!(
                        labels_vec.len() == 1,
                        "no. of args to jump instr != 1"
                    );

                    let label_str = labels_vec[0];

                    let label_start = label_start as usize;
                    let label_end = label_end as usize;

                    let all_instrs = instr_view.instrs;

                    // Iterate over the list of instrs to find the index (PC)
                    // of the instr corresponding to the label (we do this
                    // by comparing the actual label strings)
                    let pc_of_label = get_pc_of_label(instr_view, label_str);

                    if let Some(new_pc) = pc_of_label {
                        // Update `current_instr_ptr` to the PC of the label
                        current_instr_ptr = new_pc;
                        continue;
                    } else {
                        panic!("cannot find PC corresponding to label")
                    }
                } else if let Opcode::Br = op {
                    let (args_start, args_end): (u32, u32) = instr.args.into();
                    let args = get_args(instr_view, args_start, args_end);
                    assert!(
                        args.len() == 1,
                        "br instruction must only have 1 arg"
                    );
                    let arg = args[0];
                    let value_of_arg =
                        env.get(arg).expect("arg missing from env");

                    if let BrilValue::BoolVal(surrogate_bool) = value_of_arg {
                        let br_condition = bool::from(*surrogate_bool);

                        let (labels_start, labels_end): (u32, u32) =
                            instr.instr_labels.into();
                        let labels = get_labels_vec(
                            instr_view,
                            labels_start,
                            labels_end,
                        );

                        assert!(
                            labels.len() == 2,
                            "br instruction is malformed (has != 2 labels)"
                        );

                        let true_lbl = labels[0];
                        let true_pc = get_pc_of_label(instr_view, true_lbl)
                            .expect("label for true case doesn't have a PC");

                        let false_lbl = labels[1];
                        let false_pc = get_pc_of_label(instr_view, false_lbl)
                            .expect("label for false case doesn't have a PC");

                        if br_condition {
                            current_instr_ptr = true_pc;
                            continue;
                        } else {
                            current_instr_ptr = false_pc;
                            continue;
                        }
                    } else {
                        panic!(
                            "argument to br instruction is ill-typed (doesn't have type bool)"
                        );
                    }
                } else if let Opcode::Call = op {
                    interp_call(instr_view, env, funcs, instr, instr_kind);
                    current_instr_ptr += 1;
                } else if let Opcode::Ret = op {
                    let return_args = instr.args;
                    if return_args.first == -1 && return_args.second == -1 {
                        // No args supplied to Ret
                        return Ok(None);
                    }
                    let args = get_args(
                        instr_view,
                        return_args.first as u32,
                        return_args.second as u32,
                    );
                    assert!(
                        args.len() <= 1,
                        "too many args supplied to Ret instruction"
                    );

                    let arg = args[0];
                    let ret_value = env.get(arg).expect("missing arg in env");
                    return Ok(Some(*ret_value));
                } else {
                    // There are no more EffectOps to handle
                    unreachable!()
                }
            }
            InstrKind::ValueOp => {
                if op.is_binop() {
                    interp_binop(instr_view, op, instr, env);
                } else if op.is_unop() {
                    interp_unop(instr_view, op, instr, env);
                } else if let Opcode::Call = op {
                    interp_call(instr_view, env, funcs, instr, instr_kind);
                } else {
                    // there are no more ValueOps to handle
                    unreachable!()
                }
                current_instr_ptr += 1;
                continue;
            }
            InstrKind::Nop => {
                current_instr_ptr += 1;
            }
        }
    }
    Ok(None)
}

/// Interprets an entire program using the `cmd_line_args` (args to `main`)
pub fn interp_program(program: &[InstrView], cmd_line_args: Vec<&str>) {
    let mut funcs = HashMap::new();

    // Find the main function
    for view in program.iter() {
        let func_name = str::from_utf8(view.func_name).expect("invalid utf-8");
        // Remove excess null terminators at end of string
        let func_name = func_name.trim_end_matches(char::from(0));
        funcs.insert(func_name, view);
    }

    // Prepopulate the env with command line arguments
    let mut env = Environment::new();
    for (ff_arg, arg_value) in
        funcs["main"].func_args.iter().zip(cmd_line_args.iter())
    {
        let (ff_args_start, ff_args_end): (u32, u32) =
            ff_arg.arg_name_idxes.into();
        let arg_name = get_var(funcs["main"], ff_args_start, ff_args_end);
        match ff_arg.arg_type {
            FlatType::Bool => {
                if *arg_value == "true" {
                    env.insert(arg_name, BrilValue::BoolVal(true.into()));
                } else if *arg_value == "false" {
                    env.insert(arg_name, BrilValue::BoolVal(false.into()));
                }
            }
            FlatType::Int => {
                // Actually try to parse the string as an int
                let i = arg_value
                    .parse::<i64>()
                    .expect("Unable to parse string as i64");
                env.insert(arg_name, BrilValue::IntVal(i));
            }
            FlatType::Null => {
                panic!("function argument has unexpected null type");
            }
        }
    }

    interp_instr_view(funcs["main"], &mut env, &funcs)
        .expect("unexpected error when interpreting main");
}
