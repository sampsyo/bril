#![allow(dead_code)]

use crate::types::*;
use std::str;

/// Takes an `InstrStore` (flattened instrs + arrays storing args/dests etc.)
/// corresponding to a Bril function and returns its JSON representation
pub fn unflatten_instrs(instr_store: &InstrStore) -> serde_json::Value {
    let mut instr_json_vec = vec![];

    for instr in &instr_store.instrs {
        if let Some((start_idx, end_idx)) = instr.label {
            let start_idx = start_idx as usize;
            let end_idx = end_idx as usize;
            let label = &instr_store.labels_store[start_idx..=end_idx];
            let label_for_json = str::from_utf8(label).expect("invalid utf-8");
            let json = serde_json::json!({
                "label": label_for_json
            });
            instr_json_vec.push(json);
        } else {
            let op_str = Opcode::op_idx_to_op_str(instr.op as usize);

            // Extract the `ty` field of the instr as a string
            let mut ty_str = None;
            if let Some(ty) = &instr.ty {
                ty_str = Some(ty.as_str());
            }

            // Convert the `dest` index of the instr to an actual string
            // containing the dest
            let mut dest: Option<&[u8]> = None;
            if let Some((start_idx, end_idx)) = instr.dest {
                let start_idx = start_idx as usize;
                let end_idx = end_idx as usize;
                dest = Some(&instr_store.var_store[start_idx..=end_idx]);
            }

            // Convert the (start_idx, end_idx) for args in the instr to
            // an actual list of strings (by doing `args_store[start_idx..=end_idx]`)
            let mut args: Vec<&[u8]> = vec![];
            if let Some((start_idx, end_idx)) = instr.args {
                let start_idx = start_idx as usize;
                let end_idx = end_idx as usize;
                let arg_idxes: Vec<(u32, u32)> =
                    instr_store.args_idxes_store[start_idx..=end_idx].to_vec();
                for (start, end) in arg_idxes {
                    let start = start as usize;
                    let end = end as usize;
                    let arg: &[u8] = &instr_store.var_store[start..=end];
                    args.push(arg);
                }
            }

            // Convert the (start_idx, end_idx) for labels in the instr to
            // an actual list of strings
            let mut labels: Vec<&[u8]> = vec![];
            if let Some((start_idx, end_idx)) = instr.instr_labels {
                let start_idx = start_idx as usize;
                let end_idx = end_idx as usize;
                let labels_idxes: Vec<(u32, u32)> = instr_store
                    .labels_idxes_store[start_idx..=end_idx]
                    .to_vec();
                for (start, end) in labels_idxes {
                    let start = start as usize;
                    let end = end as usize;
                    let label: &[u8] = &instr_store.labels_store[start..=end];
                    labels.push(label);
                }
            }

            // Convert the (start_idx, end_idx) for funcs in the instr to
            // an actual list of strings
            let mut funcs: Option<&[u8]> = None;
            if let Some((start_idx, end_idx)) = instr.funcs {
                let start_idx = start_idx as usize;
                let end_idx = end_idx as usize;
                funcs = Some(&instr_store.funcs_store[start_idx..=end_idx]);
            }

            let args_for_json: Vec<&str> = args
                .iter()
                .map(|arg| str::from_utf8(arg).expect("invalid utf-8"))
                .collect();
            let labels_for_json: Vec<&str> = labels
                .iter()
                .map(|label| str::from_utf8(label).expect("invalid utf-8"))
                .collect();

            let mut funcs_for_json = vec![];
            if Option::is_some(&funcs) {
                let func_str = str::from_utf8(funcs.expect("missing funcs"))
                    .expect("invalid utf-8");
                funcs_for_json.push(func_str)
            }

            // Convert the `BrilValue` to an `serde_json::Value`
            let mut value_for_json: Option<serde_json::Value> = None;
            if let Some(bril_value) = &instr.value {
                match bril_value {
                    BrilValue::IntVal(i) => {
                        value_for_json = Some(serde_json::to_value(i).unwrap());
                    }
                    BrilValue::BoolVal(surrogate_bool) => {
                        value_for_json = Some(
                            serde_json::to_value(bool::from(*surrogate_bool))
                                .unwrap(),
                        );
                    }
                }
            }

            // Build a JSON object corresponding to the right instr kind
            let instr_kind = instr.get_instr_kind();
            let instr_json = match instr_kind {
                InstrKind::Label => {
                    // labels are already handled at the beginning of this function
                    unreachable!();
                }
                InstrKind::Nop => {
                    serde_json::json!({
                        "op": op_str
                    })
                }
                InstrKind::Const => {
                    let dest_for_json =
                        str::from_utf8(dest.expect("missing dest"))
                            .expect("invalid utf-8");
                    serde_json::json!({
                      "op": op_str,
                      "dest": dest_for_json,
                      "type": ty_str.expect("Expected string representing a type"),
                      "value": value_for_json.expect("Missing value"),
                    })
                }
                InstrKind::ValueOp => {
                    let dest_for_json =
                        str::from_utf8(dest.expect("missing dest"))
                            .expect("invalid utf-8");
                    serde_json::json!({
                      "op": op_str,
                      "dest": dest_for_json,
                      "type": ty_str.expect("Expected string representing a type"),
                      "args": args_for_json,
                      "labels": labels_for_json,
                      "funcs": funcs_for_json
                    })
                }
                InstrKind::EffectOp => {
                    serde_json::json!({
                      "op": op_str,
                      "args": args_for_json,
                      "labels": labels_for_json,
                      "funcs": funcs_for_json
                    })
                }
            };

            instr_json_vec.push(instr_json);
        }
    }

    // Convert the function name from raw bytes back to a UTF-8 string
    let func_name =
        str::from_utf8(&instr_store.func_name).expect("invalid utf-8");

    // Recover the arguments to the function (if any exist)
    let mut func_args_for_json = vec![];
    for func_arg in &instr_store.func_args {
        // For each arg, use its start & end index to index into the `var_store`
        // buffer, then convert those bytes back to a valid string
        let (start_idx, end_idx) = func_arg.arg_name_idxes;
        let start_idx = start_idx as usize;
        let end_idx = end_idx as usize;
        let func_arg_str =
            str::from_utf8(&instr_store.var_store[start_idx..=end_idx])
                .expect("invalid utf-8");

        // Extract the type of the function argument
        let arg_type_str = func_arg.arg_type.as_str();
        let func_arg_json = serde_json::json!({
            "name": func_arg_str,
            "type": arg_type_str
        });
        func_args_for_json.push(func_arg_json);
    }

    // Distinguish between funcs that have a return type & void functions
    // (for void functions, there is no "type" field in the JSON object
    // representing the function)
    let func_json;
    if let Some(ret_ty) = &instr_store.func_ret_ty {
        func_json = serde_json::json!({
            "name": func_name,
            "args": func_args_for_json,
            "type": ret_ty.as_str(),
            "instrs": instr_json_vec
        });
    } else {
        func_json = serde_json::json!({
            "name": func_name,
            "args": func_args_for_json,
            "instrs": instr_json_vec
        });
    }

    func_json
}
