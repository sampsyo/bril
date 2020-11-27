use serde::{Deserialize, Serialize};
use std::io::{self, Read};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Program {
    pub functions: Vec<Function>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Function {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<Vec<Argument>>,
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_type: Option<Type>,
    pub instrs: Vec<Code>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Argument {
    pub name: String,
    #[serde(rename = "type")]
    pub arg_type: Type,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum Code {
    Label { label: String },
    Instruction(Instruction),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum Instruction {
    Constant {
        op: ConstOps,
        dest: String,
        #[serde(rename = "type")]
        const_type: Type,
        value: Literal,
    },
    Value {
        op: ValueOps,
        dest: String,
        #[serde(rename = "type")]
        op_type: Type,
        #[serde(skip_serializing_if = "Option::is_none")]
        args: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        funcs: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        labels: Option<Vec<String>>,
    },
    Effect {
        op: EffectOps,
        #[serde(skip_serializing_if = "Option::is_none")]
        args: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        funcs: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        labels: Option<Vec<String>>,
    },
}
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ConstOps {
    #[serde(rename = "const")]
    Const,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum EffectOps {
    #[serde(rename = "jmp")]
    Jump,
    #[serde(rename = "br")]
    Branch,
    Call,
    #[serde(rename = "ret")]
    Return,
    Print,
    Nop,
    Store,
    Free,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ValueOps {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Lt,
    Gt,
    Le,
    Ge,
    Not,
    And,
    Or,
    Call,
    Id,
    Phi,
    Fadd,
    Fsub,
    Fmul,
    Fdiv,
    Feq,
    Flt,
    Fgt,
    Fle,
    Fge,
    Alloc,
    Load,
    PtrAdd,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Type {
    Int,
    Bool,
    Float,
    #[serde(rename = "ptr")]
    Pointer(Box<Type>),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum Literal {
    Int(i64),
    Bool(bool),
    Float(f64),
}

pub fn load_program() -> Program {
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer).unwrap();
    serde_json::from_str(&buffer).unwrap()
}
