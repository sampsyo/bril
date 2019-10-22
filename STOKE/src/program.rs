extern crate serde_json;

use serde::{Serialize, Deserialize};

use std::error::Error;
use std::fs::File;
use std::io::BufReader;

#[derive(Deserialize, Debug, EnumString)]
pub enum Op {
    #[strum(serialize = "add")]
    Add,
    #[strum(serialize = "mul")]
    Mul,
    #[strum(serialize = "sub")]
    Sub,
    #[strum(serialize = "div")]
    Div,
    #[strum(serialize = "id")]
    Id,
    #[strum(serialize = "const")]
    Const,
    #[strum(serialize = "lt")]
    Lt,
    #[strum(serialize = "le")]
    Le,
    #[strum(serialize = "gt")]
    Gt,
    #[strum(serialize = "ge")]
    Ge,
    #[strum(serialize = "eq")]
    Equal,
    #[strum(serialize = "not")]
    Not,
    #[strum(serialize = "and")]
    And,
    #[strum(serialize = "or")]
    Or,
    #[strum(serialize = "print")]
    Print,
    #[strum(serialize = "nop")]
    Nop
}

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

#[derive(Deserialize, Debug)]
pub struct Instruction {
    args: Option<Vec<String>>,
    dest: Option<String>,
    op: String,
    value: Option<i32>,
    vtype: Option<String>
}

pub fn read_json() -> Result<Program, Box<Error>>{
    let prog_file = File::open("test.json")?;
    let prog_reader = BufReader::new(prog_file);
    let prog_json = serde_json::from_reader(prog_reader)?;

    Ok(prog_json)
}