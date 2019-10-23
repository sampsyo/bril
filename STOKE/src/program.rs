use serde::{Serialize, Deserialize};

use std::error::Error;
use std::fs::File;
use std::io::BufReader;

#[derive(Deserialize, Debug)]
pub enum InstrType {
    Vint(i32),
    Vbool(bool)
}

#[derive(Deserialize, Debug)]
pub struct Program {
    pub functions: Vec<Function>,
}

#[derive(Deserialize, Debug)]
pub struct Function {
    pub instrs: Vec<Instruction>,
}

#[derive(Clone, Deserialize, Debug)]
pub struct Instruction {
    args: Option<Vec<String>>,
    dest: Option<String>,
    op: String,
    value: Option<i32>,
    vtype: Option<String>
}

impl Instruction {
    pub fn new(args: Option<Vec<String>>, dest: Option<String>, op: String, value: Option<i32>, vtype: Option<String>) -> Option<Instruction> {
        let instruction = match op.as_ref() {
            "nop" => Some(Instruction {
                args: None,
                dest: None,
                op: "nop".to_string(),
                value: None,
                vtype: None,
            }),
            _ => None
        };
        return instruction;
    }
}

pub fn read_json() -> Result<Program, Box<Error>>{
    let prog_file = File::open("test.json")?;
    let prog_reader = BufReader::new(prog_file);
    let prog_json = serde_json::from_reader(prog_reader)?;

    Ok(prog_json)
}