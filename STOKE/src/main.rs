use std::error::Error;
use std::fs::File;
use std::io::{Write};
use std::io::{Stdout};
use subprocess::{Exec};

mod search;

fn main() {
    let prog_json = search::cost::program::read_json().unwrap();
    let original_prog = &prog_json.functions[0].instrs;
    let original_cost: f32 = search::cost::cost(&original_prog);
    let mut outer_programs = Vec::new();
    let mut int_variables: Vec<String> = Vec::new();
    let mut bool_variables: Vec<String> = Vec::new();

    for instr in original_prog {
        let vtype: String = match instr.r#type.as_ref() {
            Some(x) => x.to_string(),
            None => "none".to_string(),
        };

        if (vtype == "int") {
            int_variables.push(instr.dest.clone().unwrap());
        }
        else if (vtype == "bool") {
            bool_variables.push(instr.dest.clone().unwrap());
        }
    };

    for i in 0..10 {
        println!("Iteration: {}", i);
        let mut perturb_prog = original_prog.clone();
        let mut perturb_cost = original_cost;
        let mut inner_programs = Vec::new();
        for j in 0..10 {
        let mut result = search::mc_step(&original_prog, perturb_cost, perturb_prog.as_slice(), int_variables.clone(), bool_variables.clone());
            perturb_prog = result.0;
            perturb_cost = result.1;
            inner_programs.push((perturb_prog.clone(), perturb_cost));
            let optimized_inner = search::cost::program::Program::new(search::cost::program::Function::new(perturb_prog.clone()));

        }
        inner_programs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        outer_programs.push(inner_programs[0].clone());
    }
    outer_programs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let optimized = search::cost::program::Program::new(search::cost::program::Function::new(outer_programs[0].0.clone()));
    let mut optimized_file = match File::create("original.json") {
        Err(e) => panic!("couldn't create file"),
        Ok(file) => file,
    };
    write!(optimized_file, "{:?}", serde_json::to_string(&optimized));
    println!("{:?}", serde_json::to_string(&optimized));
    println!("{:?}", outer_programs[0].1.clone());
}