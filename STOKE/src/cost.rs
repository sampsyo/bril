use std::error::Error;
use std::fs::File;
use std::io::{Write};
use std::io::{Stdout};
use subprocess::{Exec};

#[path = "program.rs"]
pub mod program;

macro_rules! new_prog {
    ($instrs:item) => {
        program::Program::new(program::Function::new($instrs))
    };
}

pub fn verify_trivial(instrs_one: &[program::Instruction], instrs_two: &[program::Instruction]) -> f32 {
    let program_one: program::Program = program::Program::new(program::Function::new(instrs_one.to_vec()));
    let program_two: program::Program = program::Program::new(program::Function::new(instrs_two.to_vec()));

    let json_one = serde_json::to_string(&program_one);
    let json_two = serde_json::to_string(&program_two);

    let mut original = match File::create("original.json") {
        Err(e) => panic!("couldn't create file"),
        Ok(file) => file,
    };
    let mut perturbed = match File::create("perturb.json") {
        Err(e) => panic!("couldn't create file"),
        Ok(file) => file,
    };

    write!(original, "{}", json_one.unwrap());
    write!(perturbed, "{}", json_two.unwrap());

    let original_result = {
        Exec::shell("cat original.json") | Exec::cmd("brili")
    }.capture().unwrap().stdout_str();
    let perturb_result = {
        Exec::shell("cat perturb.json") | Exec::cmd("brili")
    }.capture().unwrap().stdout_str();

    if (original_result == perturb_result) {
        return 1.1;
    }
    return 0.1;
}

pub fn verify_shrimp(instrs_one: &[program::Instruction], instrs_two: &[program::Instruction]) -> f32 {
    let program_one: program::Program = program::Program::new(program::Function::new(instrs_one.to_vec()));
    let program_two: program::Program = program::Program::new(program::Function::new(instrs_two.to_vec()));

    let json_one = serde_json::to_string(&program_one);
    let json_two = serde_json::to_string(&program_two);

    let mut original = match File::create("original.json") {
        Err(e) => panic!("couldn't create file"),
        Ok(file) => file,
    };
    let mut perturbed = match File::create("perturb.json") {
        Err(e) => panic!("couldn't create file"),
        Ok(file) => file,
    };

    write!(original, "{}", json_one.unwrap());
    write!(perturbed, "{}", json_two.unwrap());

    let result = {
        Exec::shell("../shrimp/shrimp --verify ./original.json ./perturb.json")
    }.capture().unwrap().stdout_str();

    if (result == "") {
        return 1.1;
    }
    return 0.1;
}

pub fn cost(instrs: &[program::Instruction]) -> f32 {
    let mut total_cost = 0.0;
    for instr in instrs.to_vec() {
        total_cost = total_cost + instr.cost().unwrap();
    };
    return (1.0 / total_cost.exp());
}