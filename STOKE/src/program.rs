use serde::{Serialize, Deserialize};

use std::error::Error;
use std::fs::File;
use std::io::BufReader;

#[derive(Deserialize, Debug, Serialize)]
pub enum InstrType {
    Vint(i32),
    Vbool(bool)
}

#[derive(Deserialize, Debug, Serialize)]
pub struct Program {
    pub functions: Vec<Function>,
}

#[derive(Deserialize, Debug, Serialize)]
pub struct Function {
    pub instrs: Vec<Instruction>,
    name: String,
}

#[derive(Clone, Deserialize, Debug, Serialize)]
pub struct Instruction {
    #[serde(skip_serializing_if = "Option::is_none")]
    args: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dest: Option<String>,
    pub op: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    value: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>
}

impl Program {
    pub fn new(function: Function) -> Program {
        let program = Program {
            functions: vec![function],
        };
        return program;
    }
}

impl Function {
    pub fn new(instrs: Vec<Instruction>) -> Function {
        let function = Function {
            instrs: instrs,
            name: "main".to_string(),
        };
        return function;
    }
}

impl Instruction {
    pub fn new(args: Option<Vec<String>>, dest: Option<String>, op: String, value: Option<i32>, vtype: Option<String>) -> Option<Instruction> {
        let instruction = match op.as_ref() {
            "nop" => Some(Instruction {
                args: Some(Vec::new()),
                dest: None,
                op: "nop".to_string(),
                value: None,
                r#type: None,
            }),
            "add" | "mul" | "sub" | "div" => Some(Instruction {
                args: args,
                dest: dest,
                op: op,
                value: None,
                r#type: Some("int".to_string()),
            }),
            "lt" | "le" | "gt" | "ge" | "eq" | "and" | "or" | "not" => Some(Instruction {
                args: args,
                dest: dest,
                op: op,
                value: None,
                r#type: Some("bool".to_string()),
            }),
            "const" => Some(Instruction {
                args: None,
                dest: dest,
                op: op,
                value: value,
                r#type: vtype,
            }),
            "id" => Some(Instruction {
                args: args,
                dest: dest,
                op: op,
                value: None,
                r#type: None,
            }),
            _ => None
        };
        return instruction;
    }

    pub fn cost(&self) -> Option<f32> {
        let instr_cost = match self.op.as_ref() {
            "const" | "id" => Some(1.0),
            "add" | "mul" | "sub" | "div" |
            "lt" | "le" | "gt" | "ge" |
            "eq" | "not" | "and" | "or" => Some(4.0),
            "nop" | "print" => Some(0.0),
            _ => Some(1000.0),
        };
        return instr_cost;
    }
}

pub fn read_json() -> Result<Program, Box<Error>>{
    let prog_file = File::open("test3.json")?;
    let prog_reader = BufReader::new(prog_file);
    let prog_json = serde_json::from_reader(prog_reader)?;

    Ok(prog_json)
}