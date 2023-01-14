use std::fmt::{self, Display, Formatter};
use std::marker::PhantomData;

use crate::{program::Literal, ConstOps};

#[cfg(feature = "position")]
use crate::program::Position;

#[cfg(feature = "import")]
use crate::program::Import;

use serde::{Deserialize, Serialize};

use serde::de::{self, Error, MapAccess, Visitor};

use serde::ser::{SerializeMap, Serializer};

/// Equivalent to a file of bril code
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AbstractProgram {
    /// A list of functions declared in the program
    pub functions: Vec<AbstractFunction>,
    /// A list of imports for this program
    #[cfg(feature = "import")]
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub imports: Vec<Import>,
}

impl Display for AbstractProgram {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for func in &self.functions {
            writeln!(f, "{func}")?;
        }
        Ok(())
    }
}

/// <https://capra.cs.cornell.edu/bril/lang/syntax.html#function>
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AbstractFunction {
    /// Any arguments the function accepts
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<AbstractArgument>,
    /// The instructions of this function
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub instrs: Vec<AbstractCode>,
    /// The name of the function
    pub name: String,
    /// The position of this function in the original source code
    #[cfg(feature = "position")]
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub pos: Option<Position>,
    /// The possible return type of this function
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_type: Option<AbstractType>,
}

impl Display for AbstractFunction {
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
pub struct AbstractArgument {
    /// a
    pub name: String,
    /// int
    #[serde(rename = "type")]
    pub arg_type: AbstractType,
}

impl Display for AbstractArgument {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name, self.arg_type)
    }
}

/// <https://capra.cs.cornell.edu/bril/lang/syntax.html#function>
/// Code is a Label or an Instruction
#[cfg_attr(not(feature = "float"), derive(Eq))]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum AbstractCode {
    /// <https://capra.cs.cornell.edu/bril/lang/syntax.html#label>
    Label {
        /// The name of the label
        label: String,
        /// Where the label is located in source code
        #[cfg(feature = "position")]
        #[serde(flatten, skip_serializing_if = "Option::is_none")]
        pos: Option<Position>,
    },
    /// <https://capra.cs.cornell.edu/bril/lang/syntax.html#instruction>
    Instruction(AbstractInstruction),
}

impl Display for AbstractCode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Label {
                label,
                #[cfg(feature = "position")]
                    pos: _,
            } => write!(f, ".{label}:"),
            Self::Instruction(instr) => write!(f, "  {instr}"),
        }
    }
}

/// <https://capra.cs.cornell.edu/bril/lang/syntax.html#instruction>
#[cfg_attr(not(feature = "float"), derive(Eq))]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum AbstractInstruction {
    /// <https://capra.cs.cornell.edu/bril/lang/syntax.html#constant>
    Constant {
        /// destination variable
        dest: String,
        /// "const"
        op: ConstOps,
        /// The source position of the instruction if provided
        #[cfg(feature = "position")]
        #[serde(flatten, skip_serializing_if = "Option::is_none")]
        pos: Option<Position>,
        /// Type of variable
        #[serde(rename = "type")]
        const_type: Option<AbstractType>,
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
        op: String,
        /// The source position of the instruction if provided
        #[cfg(feature = "position")]
        #[serde(flatten, skip_serializing_if = "Option::is_none")]
        pos: Option<Position>,
        /// Type of variable
        #[serde(rename = "type")]
        op_type: Option<AbstractType>,
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
        op: String,
        /// The source position of the instruction if provided
        #[cfg(feature = "position")]
        #[serde(flatten, skip_serializing_if = "Option::is_none")]
        pos: Option<Position>,
    },
}

impl Display for AbstractInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Constant {
                op,
                dest,
                const_type,
                value,
                #[cfg(feature = "position")]
                    pos: _,
            } => match const_type {
                Some(const_type) => write!(f, "{dest}: {const_type} = {op} {value};"),
                None => write!(f, "{dest} = {op} {value};"),
            },
            Self::Value {
                op,
                dest,
                op_type,
                args,
                funcs,
                labels,
                #[cfg(feature = "position")]
                    pos: _,
            } => {
                match op_type {
                    Some(op_type) => write!(f, "{dest}: {op_type} = {op}")?,
                    None => write!(f, "{dest} = {op}")?,
                }
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
            Self::Effect {
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

/// <https://capra.cs.cornell.edu/bril/lang/syntax.html#type>
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AbstractType {
    /// For example `bool` => `Primitive("bool")`
    Primitive(String),
    /// For example `ptr<bool>` => `Parameterized("ptr", Box::new(Primitive("bool")))`
    Parameterized(String, Box<Self>),
}

struct AbstractTypeVisitor {
    marker: PhantomData<fn() -> AbstractType>,
}

impl AbstractTypeVisitor {
    fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl<'de> Visitor<'de> for AbstractTypeVisitor {
    type Value = AbstractType;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("An AbstractType struct")
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: Error,
    {
        Ok(AbstractType::Primitive(v.to_string()))
    }

    fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(AbstractType::Primitive(v))
    }

    fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        // While there are entries remaining in the input, add them
        // into our map.
        if let Some((key, value)) = access.next_entry()? {
            Ok(AbstractType::Parameterized(key, value))
        } else {
            Err(M::Error::custom(
                "Expected one value in map for AbstractType",
            ))
        }
    }
}

impl<'de> Deserialize<'de> for AbstractType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(AbstractTypeVisitor::new())
    }
}

impl Serialize for AbstractType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Primitive(s) => serializer.serialize_str(s),
            Self::Parameterized(t, at) => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry(t, at)?;
                map.end()
            }
        }
    }
}

impl Display for AbstractType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Primitive(t) => write!(f, "{t}"),
            Self::Parameterized(t, at) => write!(f, "{t}<{at}>"),
        }
    }
}
