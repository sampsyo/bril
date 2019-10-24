extern crate rand;

use rand::Rng;
use std::collections::HashSet;

#[path = "cost.rs"]
pub mod cost;

fn remove(instructions: Vec<cost::program::Instruction>, line: usize) -> Vec<cost::program::Instruction> {
    let mut instructions_perturb = instructions.clone();
    let nop_instruction = cost::program::Instruction::new(None, None, "nop".to_string(), None, None);
    std::mem::replace(&mut instructions_perturb[line], nop_instruction.unwrap());

    return instructions_perturb;
}

fn swap(instructions: Vec<cost::program::Instruction>, line_one: usize, line_two: usize) -> Vec<cost::program::Instruction> {
    let mut instructions_perturb = instructions.clone();
    instructions_perturb.swap(line_one, line_two);

    return instructions_perturb;
}

fn replace(instructions: Vec<cost::program::Instruction>, line: usize, int_variables: Vec<String>, bool_variables: Vec<String>) -> Vec<cost::program::Instruction> {
    let mut instructions_perturb = instructions.clone();
    let mut int_variables_copy = Vec::new();
    let mut bool_variables_copy = Vec::new();

    let mut rng = rand::thread_rng();

    let mut op_codes: Vec<String> = vec![
        "add".to_string(),
        "mul".to_string(),
        "sub".to_string(),
        "div".to_string(),
        "const".to_string(),
        "id".to_string(),
        "lt".to_string(),
        "le".to_string(),
        "gt".to_string(),
        "ge".to_string(),
        "eq".to_string(),
        "and".to_string(),
        "or".to_string(),
        "not".to_string()
    ];

    let instr = instructions_perturb[line].clone();

    let vtype: String = match instr.r#type.as_ref() {
        Some(x) => x.to_string(),
        None => "none".to_string(),
    };

    for i in 0..line {
        let vtype: String = match instructions_perturb[i].r#type.as_ref() {
            Some(x) => x.to_string(),
            None => "none".to_string(),
        };

        if (vtype == "int") {
            int_variables_copy.push(instr.dest.clone().unwrap());
        }
        else if (vtype == "bool") {
            bool_variables_copy.push(instr.dest.clone().unwrap());
        }
    };

    println!("{:?}", line);
    println!("{:?}", int_variables_copy);
    println!("{:?}", bool_variables_copy);

    // let mut rand_op: String = match vtype.as_ref() {
    //     "int" => op_codes[rng.gen_range(0, 6)].clone(),
    //     "bool" => op_codes[rng.gen_range(4, 14)].clone(),
    //     _ => "nop".to_string(),
    // };

    // let rand_val: i32 = rng.gen_range(-256, 256);
    let num_args = match instr.op.as_ref() {
        "not" | "id" => 1,
        _ => 2,
    };
    let mut args: Vec<String> = Vec::new();
    if (num_args == 1) {
        if (vtype == "int" && int_variables_copy.len() > 0) {
            args.push(int_variables_copy[rng.gen_range(0, int_variables_copy.len())].clone());
        }
        else if (bool_variables_copy.len() > 0) {
            args.push(bool_variables_copy[rng.gen_range(0, bool_variables_copy.len())].clone());
        }
    }
    else {
        if (vtype == "int" && int_variables_copy.len() > 0) {
            args.push(int_variables_copy[rng.gen_range(0, int_variables_copy.len())].clone());
            args.push(int_variables_copy[rng.gen_range(0, int_variables_copy.len())].clone());
        }
        else if (bool_variables_copy.len() > 0) {
            args.push(bool_variables_copy[rng.gen_range(0, bool_variables_copy.len())].clone());
            args.push(bool_variables_copy[rng.gen_range(0, bool_variables_copy.len())].clone());
        }
    }

    if (instr.op != "nop" && instr.op != "const") {
        let new_instruction = cost::program::Instruction::new(Some(args), Some(instr.dest.unwrap()), instr.op, Some(0), Some(vtype));
        std::mem::replace(&mut instructions_perturb[line], new_instruction.unwrap());
    }

    println!("{:?}", instructions_perturb);

    return instructions_perturb;
}

pub fn mc_step(original_prog: &[cost::program::Instruction], original_score: f32, perturb_prog: &[cost::program::Instruction], int_variables: Vec<String>, bool_variables: Vec<String>) -> (Vec<cost::program::Instruction>, f32) {
    let mut rng = rand::thread_rng();
    let perturbation: i32 = rng.gen_range(0, 2);
    let line: usize = rng.gen_range(0, perturb_prog.len() - 1);
    let line2: usize = rng.gen_range(0, perturb_prog.len() - 1);

    let new_perturb_prog = match perturbation {
        0 => remove(perturb_prog.to_vec(), line),
        1 => swap(perturb_prog.to_vec(), line, line2),
        2 => replace(perturb_prog.to_vec(), line, int_variables, bool_variables),
        _ => perturb_prog.to_vec(),
    };

    let new_score: f32 = cost::verify_trivial(original_prog, &new_perturb_prog) + cost::cost(&new_perturb_prog);
    let acceptance_criteria: f32 = new_score / original_score;
    let sample: f32 = rng.gen_range(0.0, 1.0);
    if (sample > acceptance_criteria) {
        return (perturb_prog.to_vec(), original_score);
    }
    return (new_perturb_prog.to_vec(), new_score);
}