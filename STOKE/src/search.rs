extern crate rand;

use rand::Rng;

#[path = "program.rs"]
pub mod program;

fn remove(instructions: Vec<program::Instruction>, line: usize) -> Vec<program::Instruction> {
    let mut instructions_perturb = instructions.clone();
    let nop_instruction = program::Instruction::new(None, None, "nop".to_string(), None, None);
    std::mem::replace(&mut instructions_perturb[line], nop_instruction.unwrap());

    return instructions_perturb;
}

fn swap(instructions: Vec<program::Instruction>, line_one: usize, line_two: usize) -> Vec<program::Instruction> {
    let mut instructions_perturb = instructions.clone();
    instructions_perturb.swap(line_one, line_two);

    return instructions_perturb;
}

fn replace(instructions: Vec<program::Instruction>, line: usize) -> Vec<program::Instruction> {
    let mut instructions_perturb = instructions.clone();

    return instructions_perturb;
}

pub fn mc_step(original_prog: &[program::Instruction]) -> Vec<program::Instruction> {
    let mut rng = rand::thread_rng();
    let perturbation: i32 = rng.gen_range(0, 2);
    let line: usize = rng.gen_range(0, original_prog.len());
    let line2: usize = rng.gen_range(0, original_prog.len());

    let new_prog = match perturbation {
        0 => remove(original_prog.to_vec(), line),
        1 => swap(original_prog.to_vec(), line, line2),
        _ => original_prog.to_vec(),
    };

    return new_prog;
}
