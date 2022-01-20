use std::fmt::{self, Display, Formatter};
use std::marker::PhantomData;

use crate::{program::Literal, ConstOps};

use serde::{Deserialize, Serialize};

use serde::de::{self, Error, MapAccess, Visitor};

use serde::ser::{SerializeMap, Serializer};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AbstractProgram {
    pub functions: Vec<AbstractFunction>,
}

impl Display for AbstractProgram {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for func in &self.functions {
            writeln!(f, "{}", func)?;
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AbstractFunction {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<AbstractArgument>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub instrs: Vec<AbstractCode>,
    pub name: String,
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
pub struct AbstractArgument {
    pub name: String,
    #[serde(rename = "type")]
    pub arg_type: AbstractType,
}

impl Display for AbstractArgument {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name, self.arg_type)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum AbstractCode {
    Label { label: String },
    Instruction(AbstractInstruction),
}

impl Display for AbstractCode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            AbstractCode::Label { label } => write!(f, ".{}:", label),
            AbstractCode::Instruction(instr) => write!(f, "  {}", instr),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum AbstractInstruction {
    Constant {
        dest: String,
        op: ConstOps,
        #[serde(rename = "type")]
        const_type: AbstractType,
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
        op: String,
        #[serde(rename = "type")]
        op_type: AbstractType,
    },
    Effect {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        args: Vec<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        funcs: Vec<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        labels: Vec<String>,
        op: String,
    },
}

impl Display for AbstractInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            AbstractInstruction::Constant {
                op,
                dest,
                const_type,
                value,
            } => {
                write!(f, "{}: {} = {} {};", dest, const_type, op, value)
            }
            AbstractInstruction::Value {
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
            AbstractInstruction::Effect {
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AbstractType {
    Primitive(String),
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
            AbstractType::Primitive(s) => serializer.serialize_str(s),
            AbstractType::Parameterized(t, at) => {
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
            AbstractType::Primitive(t) => write!(f, "{}", t),

            AbstractType::Parameterized(t, at) => write!(f, "{}<{}>", t, at),
        }
    }
}
