use std::fmt::{self, Display, Formatter};

use serde::{Deserialize, Serialize};

/// Equivalent to a file of bril code
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Program {
    /// A list of functions declared in the program
    pub functions: Vec<Function>,
    #[cfg(feature = "import")]
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    /// A list of imports for this program
    pub imports: Vec<Import>,
}

impl Display for Program {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        #[cfg(feature = "import")]
        for i in &self.imports {
            writeln!(f, "{i}")?;
        }
        for func in &self.functions {
            writeln!(f, "{func}")?;
        }
        Ok(())
    }
}

/// <https://capra.cs.cornell.edu/bril/lang/import.html#syntax>
#[cfg(feature = "import")]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Import {
    /// A list of functions to be imported
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub functions: Vec<ImportedFunction>,
    /// The relative path of the file from some lib directory specified by the user
    pub path: std::path::PathBuf,
}

#[cfg(feature = "import")]
impl Display for Import {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "from {}", self.path.display())?;
        if !self.functions.is_empty() {
            write!(f, " import ")?;
            for (i, name) in self.functions.iter().enumerate() {
                if i != 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{name}")?;
            }
        }
        write!(f, ";")?;
        Ok(())
    }
}

/// <https://capra.cs.cornell.edu/bril/lang/import.html#syntax>
#[cfg(feature = "import")]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ImportedFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    /// A function can be optionally aliased with a different name for use in the rest of the program
    pub alias: Option<String>,
    /// The name of the function being imported
    pub name: String,
}

#[cfg(feature = "import")]
impl Display for ImportedFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)?;
        if let Some(a) = self.alias.as_ref() {
            write!(f, " as {}", a)?;
        }
        Ok(())
    }
}

/// <https://capra.cs.cornell.edu/bril/lang/syntax.html#function>
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Function {
    /// Any arguments the function accepts
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<Argument>,
    /// The instructions of this function
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub instrs: Vec<Code>,
    /// The name of the function
    pub name: String,
    /// The position of this function in the original source code
    #[cfg(feature = "position")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pos: Option<Position>,
    /// The possible return type of this function
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
                write!(f, "{arg}")?;
            }
            write!(f, ")")?;
        }
        if let Some(tpe) = self.return_type.as_ref() {
            write!(f, ": {tpe}")?;
        }
        writeln!(f, " {{")?;
        for instr in &self.instrs {
            writeln!(f, "{instr}")?;
        }
        write!(f, "}}")?;
        Ok(())
    }
}

/// An argument of a function
/// <https://capra.cs.cornell.edu/bril/lang/syntax.html#function>
/// Example: a : int
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Argument {
    /// a
    pub name: String,
    #[serde(rename = "type")]
    /// int
    pub arg_type: Type,
}

impl Display for Argument {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name, self.arg_type)
    }
}

/// <https://capra.cs.cornell.edu/bril/lang/syntax.html#function>
/// Code is a Label or an Instruction
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum Code {
    /// <https://capra.cs.cornell.edu/bril/lang/syntax.html#label>
    Label {
        /// The name of the label
        label: String,
        /// Where the label is located in source code
        #[cfg(feature = "position")]
        #[serde(skip_serializing_if = "Option::is_none")]
        pos: Option<Position>,
    },
    /// <https://capra.cs.cornell.edu/bril/lang/syntax.html#instruction>
    Instruction(Instruction),
}

impl Display for Code {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Code::Label {
                label,
                #[cfg(feature = "position")]
                    pos: _,
            } => write!(f, ".{label}:"),
            Code::Instruction(instr) => write!(f, "  {instr}"),
        }
    }
}

/// <https://capra.cs.cornell.edu/bril/lang/syntax.html#instruction>
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum Instruction {
    /// <https://capra.cs.cornell.edu/bril/lang/syntax.html#constant>
    Constant {
        /// destination variable
        dest: String,
        /// "const"
        op: ConstOps,
        #[cfg(feature = "position")]
        /// The source position of the instruction if provided
        #[serde(skip_serializing_if = "Option::is_none")]
        pos: Option<Position>,
        /// Type of variable
        #[serde(rename = "type")]
        const_type: Type,
        /// The literal being stored in the variable
        value: Literal,
    },
    /// <https://capra.cs.cornell.edu/bril/lang/syntax.html#value-operation>
    Value {
        /// List of variables as arguments
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        args: Vec<String>,
        /// destination variable
        dest: String,
        /// List of strings as function names
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        funcs: Vec<String>,
        /// List of strings as labels
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        labels: Vec<String>,
        /// Operation being executed
        op: ValueOps,
        /// The source position of the instruction if provided
        #[cfg(feature = "position")]
        #[serde(skip_serializing_if = "Option::is_none")]
        pos: Option<Position>,
        /// Type of variable
        #[serde(rename = "type")]
        op_type: Type,
    },
    /// <https://capra.cs.cornell.edu/bril/lang/syntax.html#effect-operation>
    Effect {
        /// List of variables as arguments
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        args: Vec<String>,
        /// List of strings as function names
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        funcs: Vec<String>,
        /// List of strings as labels
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        labels: Vec<String>,
        /// Operation being executed
        op: EffectOps,
        /// The source position of the instruction if provided
        #[cfg(feature = "position")]
        #[serde(skip_serializing_if = "Option::is_none")]
        pos: Option<Position>,
    },
}

#[cfg(feature = "position")]
impl Instruction {
    /// A helper function to extract the position value if it exists from an instruction
    #[must_use]
    pub const fn get_pos(&self) -> Option<Position> {
        match self {
            Instruction::Constant { pos, .. }
            | Instruction::Value { pos, .. }
            | Instruction::Effect { pos, .. } => *pos,
        }
    }
}

impl Display for Instruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Constant {
                op,
                dest,
                const_type,
                value,
                #[cfg(feature = "position")]
                    pos: _,
            } => {
                write!(f, "{dest}: {const_type} = {op} {value};")
            }
            Instruction::Value {
                op,
                dest,
                op_type,
                args,
                funcs,
                labels,
                #[cfg(feature = "position")]
                    pos: _,
            } => {
                write!(f, "{dest}: {op_type} = {op}")?;
                for func in funcs {
                    write!(f, " @{func}")?;
                }
                for arg in args {
                    write!(f, " {arg}")?;
                }
                for label in labels {
                    write!(f, " .{label}")?;
                }
                write!(f, ";")
            }
            Instruction::Effect {
                op,
                args,
                funcs,
                labels,
                #[cfg(feature = "position")]
                    pos: _,
            } => {
                write!(f, "{op}")?;
                for func in funcs {
                    write!(f, " @{func}")?;
                }
                for arg in args {
                    write!(f, " {arg}")?;
                }
                for label in labels {
                    write!(f, " .{label}")?;
                }
                write!(f, ";")
            }
        }
    }
}

/// <https://capra.cs.cornell.edu/bril/lang/syntax.html#constant>
#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ConstOps {
    /// "const"
    #[serde(rename = "const")]
    Const,
}

impl Display for ConstOps {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Const => write!(f, "const"),
        }
    }
}

/// <https://capra.cs.cornell.edu/bril/lang/syntax.html#effect-operation>
#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum EffectOps {
    /// <https://capra.cs.cornell.edu/bril/lang/core.html#control>
    #[serde(rename = "jmp")]
    Jump,
    /// <https://capra.cs.cornell.edu/bril/lang/core.html#control>
    #[serde(rename = "br")]
    Branch,
    /// <https://capra.cs.cornell.edu/bril/lang/core.html#control>
    Call,
    /// <https://capra.cs.cornell.edu/bril/lang/core.html#control>
    #[serde(rename = "ret")]
    Return,
    /// <https://capra.cs.cornell.edu/bril/lang/core.html#miscellaneous>
    Print,
    /// <https://capra.cs.cornell.edu/bril/lang/core.html#miscellaneous>
    Nop,
    /// <https://capra.cs.cornell.edu/bril/lang/memory.html#operations>
    #[cfg(feature = "memory")]
    Store,
    /// <https://capra.cs.cornell.edu/bril/lang/memory.html#operations>
    #[cfg(feature = "memory")]
    Free,
    /// <https://capra.cs.cornell.edu/bril/lang/spec.html#operations>
    #[cfg(feature = "speculate")]
    Speculate,
    /// <https://capra.cs.cornell.edu/bril/lang/spec.html#operations>
    #[cfg(feature = "speculate")]
    Commit,
    /// <https://capra.cs.cornell.edu/bril/lang/spec.html#operations>
    #[cfg(feature = "speculate")]
    Guard,
}

impl Display for EffectOps {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Jump => write!(f, "jmp"),
            Self::Branch => write!(f, "br"),
            Self::Call => write!(f, "call"),
            Self::Return => write!(f, "ret"),
            Self::Print => write!(f, "print"),
            Self::Nop => write!(f, "nop"),
            #[cfg(feature = "memory")]
            Self::Store => write!(f, "store"),
            #[cfg(feature = "memory")]
            Self::Free => write!(f, "free"),
            #[cfg(feature = "speculate")]
            Self::Speculate => write!(f, "speculate"),
            #[cfg(feature = "speculate")]
            Self::Commit => write!(f, "commit"),
            #[cfg(feature = "speculate")]
            Self::Guard => write!(f, "guard"),
        }
    }
}

/// <https://capra.cs.cornell.edu/bril/lang/syntax.html#value-operation>
#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum ValueOps {
    /// <https://capra.cs.cornell.edu/bril/lang/core.html#arithmetic>
    Add,
    /// <https://capra.cs.cornell.edu/bril/lang/core.html#arithmetic>
    Sub,
    /// <https://capra.cs.cornell.edu/bril/lang/core.html#arithmetic>
    Mul,
    /// <https://capra.cs.cornell.edu/bril/lang/core.html#arithmetic>
    Div,
    /// <https://capra.cs.cornell.edu/bril/lang/core.html#comparison>
    Eq,
    /// <https://capra.cs.cornell.edu/bril/lang/core.html#comparison>
    Lt,
    /// <https://capra.cs.cornell.edu/bril/lang/core.html#comparison>
    Gt,
    /// <https://capra.cs.cornell.edu/bril/lang/core.html#comparison>
    Le,
    /// <https://capra.cs.cornell.edu/bril/lang/core.html#comparison>
    Ge,
    /// <https://capra.cs.cornell.edu/bril/lang/core.html#logic>
    Not,
    /// <https://capra.cs.cornell.edu/bril/lang/core.html#logic>
    And,
    /// <https://capra.cs.cornell.edu/bril/lang/core.html#logic>
    Or,
    /// <https://capra.cs.cornell.edu/bril/lang/core.html#control>
    Call,
    /// <https://capra.cs.cornell.edu/bril/lang/core.html#miscellaneous>
    Id,
    /// <https://capra.cs.cornell.edu/bril/lang/ssa.html#operations>
    #[cfg(feature = "ssa")]
    Phi,
    /// <https://capra.cs.cornell.edu/bril/lang/float.html#operations>
    #[cfg(feature = "float")]
    Fadd,
    /// <https://capra.cs.cornell.edu/bril/lang/float.html#operations>
    #[cfg(feature = "float")]
    Fsub,
    /// <https://capra.cs.cornell.edu/bril/lang/float.html#operations>
    #[cfg(feature = "float")]
    Fmul,
    /// <https://capra.cs.cornell.edu/bril/lang/float.html#operations>
    #[cfg(feature = "float")]
    Fdiv,
    /// <https://capra.cs.cornell.edu/bril/lang/float.html#operations>
    #[cfg(feature = "float")]
    Feq,
    /// <https://capra.cs.cornell.edu/bril/lang/float.html#operations>
    #[cfg(feature = "float")]
    Flt,
    /// <https://capra.cs.cornell.edu/bril/lang/float.html#operations>
    #[cfg(feature = "float")]
    Fgt,
    /// <https://capra.cs.cornell.edu/bril/lang/float.html#operations>
    #[cfg(feature = "float")]
    Fle,
    /// <https://capra.cs.cornell.edu/bril/lang/float.html#operations>
    #[cfg(feature = "float")]
    Fge,
    /// <https://capra.cs.cornell.edu/bril/lang/memory.html#operations>
    #[cfg(feature = "memory")]
    Alloc,
    /// <https://capra.cs.cornell.edu/bril/lang/memory.html#operations>
    #[cfg(feature = "memory")]
    Load,
    /// <https://capra.cs.cornell.edu/bril/lang/memory.html#operations>
    #[cfg(feature = "memory")]
    PtrAdd,
}

impl Display for ValueOps {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Add => write!(f, "add"),
            Self::Sub => write!(f, "sub"),
            Self::Mul => write!(f, "mul"),
            Self::Div => write!(f, "div"),
            Self::Eq => write!(f, "eq"),
            Self::Lt => write!(f, "lt"),
            Self::Gt => write!(f, "gt"),
            Self::Le => write!(f, "le"),
            Self::Ge => write!(f, "ge"),
            Self::Not => write!(f, "not"),
            Self::And => write!(f, "and"),
            Self::Or => write!(f, "or"),
            Self::Call => write!(f, "call"),
            Self::Id => write!(f, "id"),
            #[cfg(feature = "ssa")]
            Self::Phi => write!(f, "phi"),
            #[cfg(feature = "float")]
            Self::Fadd => write!(f, "fadd"),
            #[cfg(feature = "float")]
            Self::Fsub => write!(f, "fsub"),
            #[cfg(feature = "float")]
            Self::Fmul => write!(f, "fmul"),
            #[cfg(feature = "float")]
            Self::Fdiv => write!(f, "fdiv"),
            #[cfg(feature = "float")]
            Self::Feq => write!(f, "feq"),
            #[cfg(feature = "float")]
            Self::Flt => write!(f, "flt"),
            #[cfg(feature = "float")]
            Self::Fgt => write!(f, "fgt"),
            #[cfg(feature = "float")]
            Self::Fle => write!(f, "fle"),
            #[cfg(feature = "float")]
            Self::Fge => write!(f, "fge"),
            #[cfg(feature = "memory")]
            Self::Alloc => write!(f, "alloc"),
            #[cfg(feature = "memory")]
            Self::Load => write!(f, "load"),
            #[cfg(feature = "memory")]
            Self::PtrAdd => write!(f, "ptradd"),
        }
    }
}

/// <https://capra.cs.cornell.edu/bril/lang/syntax.html#type>
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Type {
    /// <https://capra.cs.cornell.edu/bril/lang/core.html#types>
    Int,
    /// <https://capra.cs.cornell.edu/bril/lang/core.html#types>
    Bool,
    /// <https://capra.cs.cornell.edu/bril/lang/float.html#types>
    #[cfg(feature = "float")]
    Float,
    /// <https://capra.cs.cornell.edu/bril/lang/memory.html#types>
    #[cfg(feature = "memory")]
    #[serde(rename = "ptr")]
    Pointer(Box<Self>),
}

impl Display for Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int => write!(f, "int"),
            Self::Bool => write!(f, "bool"),
            #[cfg(feature = "float")]
            Self::Float => write!(f, "float"),
            #[cfg(feature = "memory")]
            Self::Pointer(tpe) => write!(f, "ptr<{tpe}>"),
        }
    }
}

/// A JSON number/value
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum Literal {
    /// Integers
    Int(i64),
    /// Booleans
    Bool(bool),
    /// Floating Points
    #[cfg(feature = "float")]
    Float(f64),
}

impl Display for Literal {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Int(i) => write!(f, "{i}"),
            Literal::Bool(b) => write!(f, "{b}"),
            #[cfg(feature = "float")]
            Literal::Float(x) => write!(f, "{x}"),
        }
    }
}

impl Literal {
    /// A helper function to get the type of literal values
    #[must_use]
    pub const fn get_type(&self) -> Type {
        match self {
            Literal::Int(_) => Type::Int,
            Literal::Bool(_) => Type::Bool,
            #[cfg(feature = "float")]
            Literal::Float(_) => Type::Float,
        }
    }
}

/// <https://capra.cs.cornell.edu/bril/lang/syntax.html#source-positions>
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct Position {
    /// Column
    pub col: u64,
    /// Row
    pub row: u64,
}
