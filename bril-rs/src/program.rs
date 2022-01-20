use std::fmt::{self, Display, Formatter};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Program {
    pub functions: Vec<Function>,
}

impl Display for Program {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for func in &self.functions {
            writeln!(f, "{}", func)?;
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Function {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<Argument>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub instrs: Vec<Code>,
    pub name: String,
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_type: Option<Type>,
}

impl Display for Function {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "@{}", self.name)?;
        if !self.args.is_empty() {
            write!(f, "(")?;
            for (i, arg) in self.args.iter().enumerate() {
                if i != 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", arg)?;
            }
            write!(f, ")")?;
        }
        if let Some(tpe) = self.return_type.as_ref() {
            write!(f, ": {}", tpe)?;
        }
        writeln!(f, " {{")?;
        for instr in &self.instrs {
            writeln!(f, "{}", instr)?;
        }
        write!(f, "}}")?;
        Ok(())
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Argument {
    pub name: String,
    #[serde(rename = "type")]
    pub arg_type: Type,
}

impl Display for Argument {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name, self.arg_type)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum Code {
    Label { label: String },
    Instruction(Instruction),
}

impl Display for Code {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Code::Label { label } => write!(f, ".{}:", label),
            Code::Instruction(instr) => write!(f, "  {}", instr),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum Instruction {
    Constant {
        dest: String,
        op: ConstOps,
        #[serde(rename = "type")]
        const_type: Type,
        value: Literal,
    },
    Value {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        args: Vec<String>,
        dest: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        funcs: Vec<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        labels: Vec<String>,
        op: ValueOps,
        #[serde(rename = "type")]
        op_type: Type,
    },
    Effect {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        args: Vec<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        funcs: Vec<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        labels: Vec<String>,
        op: EffectOps,
    },
}

impl Display for Instruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Constant {
                op,
                dest,
                const_type,
                value,
            } => {
                write!(f, "{}: {} = {} {};", dest, const_type, op, value)
            }
            Instruction::Value {
                op,
                dest,
                op_type,
                args,
                funcs,
                labels,
            } => {
                write!(f, "{}: {} = {}", dest, op_type, op)?;
                for func in funcs {
                    write!(f, " @{}", func)?;
                }
                for arg in args {
                    write!(f, " {}", arg)?;
                }
                for label in labels {
                    write!(f, " .{}", label)?;
                }
                write!(f, ";")
            }
            Instruction::Effect {
                op,
                args,
                funcs,
                labels,
            } => {
                write!(f, "{}", op)?;
                for func in funcs {
                    write!(f, " @{}", func)?;
                }
                for arg in args {
                    write!(f, " {}", arg)?;
                }
                for label in labels {
                    write!(f, " .{}", label)?;
                }
                write!(f, ";")
            }
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ConstOps {
    #[serde(rename = "const")]
    Const,
}

impl Display for ConstOps {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ConstOps::Const => write!(f, "const"),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq, Eq, Hash)]
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
    #[cfg(feature = "memory")]
    Store,
    #[cfg(feature = "memory")]
    Free,
    #[cfg(feature = "speculate")]
    Speculate,
    #[cfg(feature = "speculate")]
    Commit,
    #[cfg(feature = "speculate")]
    Guard,
}

impl Display for EffectOps {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            EffectOps::Jump => write!(f, "jmp"),
            EffectOps::Branch => write!(f, "br"),
            EffectOps::Call => write!(f, "call"),
            EffectOps::Return => write!(f, "ret"),
            EffectOps::Print => write!(f, "print"),
            EffectOps::Nop => write!(f, "nop"),
            #[cfg(feature = "memory")]
            EffectOps::Store => write!(f, "store"),
            #[cfg(feature = "memory")]
            EffectOps::Free => write!(f, "free"),
            #[cfg(feature = "speculate")]
            EffectOps::Speculate => write!(f, "speculate"),
            #[cfg(feature = "speculate")]
            EffectOps::Commit => write!(f, "commit"),
            #[cfg(feature = "speculate")]
            EffectOps::Guard => write!(f, "guard"),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq, Eq, Hash)]
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
    #[cfg(feature = "ssa")]
    Phi,
    #[cfg(feature = "float")]
    Fadd,
    #[cfg(feature = "float")]
    Fsub,
    #[cfg(feature = "float")]
    Fmul,
    #[cfg(feature = "float")]
    Fdiv,
    #[cfg(feature = "float")]
    Feq,
    #[cfg(feature = "float")]
    Flt,
    #[cfg(feature = "float")]
    Fgt,
    #[cfg(feature = "float")]
    Fle,
    #[cfg(feature = "float")]
    Fge,
    #[cfg(feature = "memory")]
    Alloc,
    #[cfg(feature = "memory")]
    Load,
    #[cfg(feature = "memory")]
    PtrAdd,
}

impl Display for ValueOps {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ValueOps::Add => write!(f, "add"),
            ValueOps::Sub => write!(f, "sub"),
            ValueOps::Mul => write!(f, "mul"),
            ValueOps::Div => write!(f, "div"),
            ValueOps::Eq => write!(f, "eq"),
            ValueOps::Lt => write!(f, "lt"),
            ValueOps::Gt => write!(f, "gt"),
            ValueOps::Le => write!(f, "le"),
            ValueOps::Ge => write!(f, "ge"),
            ValueOps::Not => write!(f, "not"),
            ValueOps::And => write!(f, "and"),
            ValueOps::Or => write!(f, "or"),
            ValueOps::Call => write!(f, "call"),
            ValueOps::Id => write!(f, "id"),
            #[cfg(feature = "ssa")]
            ValueOps::Phi => write!(f, "phi"),
            #[cfg(feature = "float")]
            ValueOps::Fadd => write!(f, "fadd"),
            #[cfg(feature = "float")]
            ValueOps::Fsub => write!(f, "fsub"),
            #[cfg(feature = "float")]
            ValueOps::Fmul => write!(f, "fmul"),
            #[cfg(feature = "float")]
            ValueOps::Fdiv => write!(f, "fdiv"),
            #[cfg(feature = "float")]
            ValueOps::Feq => write!(f, "feq"),
            #[cfg(feature = "float")]
            ValueOps::Flt => write!(f, "flt"),
            #[cfg(feature = "float")]
            ValueOps::Fgt => write!(f, "fgt"),
            #[cfg(feature = "float")]
            ValueOps::Fle => write!(f, "fle"),
            #[cfg(feature = "float")]
            ValueOps::Fge => write!(f, "fge"),
            #[cfg(feature = "memory")]
            ValueOps::Alloc => write!(f, "alloc"),
            #[cfg(feature = "memory")]
            ValueOps::Load => write!(f, "load"),
            #[cfg(feature = "memory")]
            ValueOps::PtrAdd => write!(f, "ptradd"),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Type {
    Int,
    Bool,
    #[cfg(feature = "float")]
    Float,
    #[cfg(feature = "memory")]
    #[serde(rename = "ptr")]
    Pointer(Box<Self>),
}

impl Display for Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Type::Int => write!(f, "int"),
            Type::Bool => write!(f, "bool"),
            #[cfg(feature = "float")]
            Type::Float => write!(f, "float"),
            #[cfg(feature = "memory")]
            Type::Pointer(tpe) => write!(f, "ptr<{}>", tpe),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum Literal {
    Int(i64),
    Bool(bool),
    #[cfg(feature = "float")]
    Float(f64),
}

impl Display for Literal {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Int(i) => write!(f, "{}", i),
            Literal::Bool(b) => write!(f, "{}", b),
            #[cfg(feature = "float")]
            Literal::Float(x) => write!(f, "{}", x),
        }
    }
}

impl Literal {
    pub const fn get_type(&self) -> Type {
        match self {
            Literal::Int(_) => Type::Int,
            Literal::Bool(_) => Type::Bool,
            #[cfg(feature = "float")]
            Literal::Float(_) => Type::Float,
        }
    }
}
