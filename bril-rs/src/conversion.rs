use std::fmt::Display;

use crate::{
    AbstractArgument, AbstractCode, AbstractFunction, AbstractInstruction, AbstractProgram,
    AbstractType, Argument, Code, EffectOps, Function, Instruction, Position, Program, Type,
    ValueOps,
};

use thiserror::Error;

// This is a nifty trick to supply a global value for pos when it is not defined
#[cfg(not(feature = "position"))]
#[allow(non_upper_case_globals)]
const pos: Option<Position> = None;

/// This is the [`std::error::Error`] implementation for `bril_rs`. This crate currently only supports errors from converting between [`AbstractProgram`] and [Program]
// todo Should this also wrap Serde errors? In this case, maybe change the name from ConversionError
// Having the #[error(...)] for all variants derives the Display trait as well
#[derive(Error, Debug)]
#[allow(clippy::module_name_repetitions)]
pub enum ConversionError {
    /// Expected a primitive type like int or bool, found {0}"
    #[error("Expected a primitive type like int or bool, found {0}")]
    InvalidPrimitive(String),

    /// Expected a parameterized type like ptr, found {0}<{1}>
    #[error("Expected a parameterized type like ptr, found {0}<{1}>")]
    InvalidParameterized(String, String),

    /// Expected an value operation, found {0}
    #[error("Expected an value operation, found {0}")]
    InvalidValueOps(String),

    /// Expected an effect operation, found {0}
    #[error("Expected an effect operation, found {0}")]
    InvalidEffectOps(String),

    /// Missing type signature
    #[error("Missing type signature")]
    MissingType,
}

impl ConversionError {
    #[doc(hidden)]
    #[must_use]
    pub const fn add_pos(self, pos_var: Option<Position>) -> PositionalConversionError {
        PositionalConversionError {
            e: self,
            pos: pos_var,
        }
    }
}

/// Wraps [`ConversionError`] to optionally provide source code positions if they are available.
#[derive(Error, Debug)]
pub struct PositionalConversionError {
    #[doc(hidden)]
    pub e: ConversionError,
    #[doc(hidden)]
    pub pos: Option<Position>,
}

impl PositionalConversionError {
    #[doc(hidden)]
    #[must_use]
    pub const fn new(e: ConversionError) -> Self {
        Self { e, pos: None }
    }
}

impl Display for PositionalConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "position")]
            Self { e, pos: Some(pos) } => {
                write!(f, "Line {}, Column {}: {e}", pos.pos.row, pos.pos.col)
            }
            #[cfg(not(feature = "position"))]
            Self { e: _, pos: Some(_) } => {
                unreachable!()
            }
            Self { e, pos: None } => write!(f, "{e}"),
        }
    }
}

impl TryFrom<AbstractProgram> for Program {
    type Error = PositionalConversionError;
    fn try_from(
        AbstractProgram {
            #[cfg(feature = "import")]
            imports,
            functions,
        }: AbstractProgram,
    ) -> Result<Self, Self::Error> {
        Ok(Self {
            #[cfg(feature = "import")]
            imports,
            functions: functions
                .into_iter()
                .map(std::convert::TryInto::try_into)
                .collect::<Result<Vec<Function>, _>>()?,
        })
    }
}

impl TryFrom<AbstractFunction> for Function {
    type Error = PositionalConversionError;
    fn try_from(
        AbstractFunction {
            args,
            instrs,
            name,
            return_type,
            #[cfg(feature = "position")]
            pos,
        }: AbstractFunction,
    ) -> Result<Self, Self::Error> {
        Ok(Self {
            args: args
                .into_iter()
                .map(std::convert::TryInto::try_into)
                .collect::<Result<Vec<Argument>, _>>()
                .map_err(|e| e.add_pos(pos.clone()))?,
            instrs: instrs
                .into_iter()
                .map(std::convert::TryInto::try_into)
                .collect::<Result<Vec<Code>, _>>()?,
            name,
            return_type: match return_type {
                None => None,
                Some(t) => Some(
                    t.try_into()
                        .map_err(|e: ConversionError| e.add_pos(pos.clone()))?,
                ),
            },
            #[cfg(feature = "position")]
            pos,
        })
    }
}

impl TryFrom<AbstractArgument> for Argument {
    type Error = ConversionError;
    fn try_from(
        AbstractArgument { name, arg_type }: AbstractArgument,
    ) -> Result<Self, Self::Error> {
        Ok(Self {
            name,
            arg_type: arg_type.try_into()?,
        })
    }
}

impl TryFrom<AbstractCode> for Code {
    type Error = PositionalConversionError;
    fn try_from(c: AbstractCode) -> Result<Self, Self::Error> {
        Ok(match c {
            AbstractCode::Label {
                label,
                #[cfg(feature = "position")]
                pos,
            } => Self::Label {
                label,
                #[cfg(feature = "position")]
                pos,
            },
            AbstractCode::Instruction(i) => Self::Instruction(i.try_into()?),
        })
    }
}

impl TryFrom<AbstractInstruction> for Instruction {
    type Error = PositionalConversionError;
    fn try_from(i: AbstractInstruction) -> Result<Self, Self::Error> {
        Ok(match i {
            AbstractInstruction::Constant {
                dest,
                op,
                const_type,
                value,
                #[cfg(feature = "position")]
                pos,
            } => Self::Constant {
                dest,
                op,
                const_type: const_type
                    .try_into()
                    .map_err(|e: ConversionError| e.add_pos(pos.clone()))?,
                value,
                #[cfg(feature = "position")]
                pos,
            },
            AbstractInstruction::Value {
                args,
                dest,
                funcs,
                labels,
                op,
                op_type,
                #[cfg(feature = "position")]
                pos,
            } => Self::Value {
                args,
                dest,
                funcs,
                labels,
                op_type: op_type
                    .try_into()
                    .map_err(|e: ConversionError| e.add_pos(pos.clone()))?,
                #[cfg(feature = "position")]
                pos: pos.clone(),
                op: match op.as_ref() {
                    "add" => ValueOps::Add,
                    "mul" => ValueOps::Mul,
                    "div" => ValueOps::Div,
                    "eq" => ValueOps::Eq,
                    "lt" => ValueOps::Lt,
                    "gt" => ValueOps::Gt,
                    "le" => ValueOps::Le,
                    "ge" => ValueOps::Ge,
                    "not" => ValueOps::Not,
                    "and" => ValueOps::And,
                    "or" => ValueOps::Or,
                    "call" => ValueOps::Call,
                    "id" => ValueOps::Id,
                    "sub" => ValueOps::Sub,
                    #[cfg(feature = "ssa")]
                    "phi" => ValueOps::Phi,
                    #[cfg(feature = "float")]
                    "fadd" => ValueOps::Fadd,
                    #[cfg(feature = "float")]
                    "fsub" => ValueOps::Fsub,
                    #[cfg(feature = "float")]
                    "fmul" => ValueOps::Fmul,
                    #[cfg(feature = "float")]
                    "fdiv" => ValueOps::Fdiv,
                    #[cfg(feature = "float")]
                    "feq" => ValueOps::Feq,
                    #[cfg(feature = "float")]
                    "flt" => ValueOps::Flt,
                    #[cfg(feature = "float")]
                    "fgt" => ValueOps::Fgt,
                    #[cfg(feature = "float")]
                    "fle" => ValueOps::Fle,
                    #[cfg(feature = "float")]
                    "fge" => ValueOps::Fge,
                    #[cfg(feature = "char")]
                    "ceq" => ValueOps::Ceq,
                    #[cfg(feature = "char")]
                    "clt" => ValueOps::Clt,
                    #[cfg(feature = "char")]
                    "cgt" => ValueOps::Cgt,
                    #[cfg(feature = "char")]
                    "cle" => ValueOps::Cle,
                    #[cfg(feature = "char")]
                    "cge" => ValueOps::Cge,
                    #[cfg(feature = "char")]
                    "char2int" => ValueOps::Char2int,
                    #[cfg(feature = "char")]
                    "int2char" => ValueOps::Int2char,
                    #[cfg(feature = "memory")]
                    "alloc" => ValueOps::Alloc,
                    #[cfg(feature = "memory")]
                    "load" => ValueOps::Load,
                    #[cfg(feature = "memory")]
                    "ptradd" => ValueOps::PtrAdd,
                    v => {
                        return Err(ConversionError::InvalidValueOps(v.to_string()))
                            .map_err(|e| e.add_pos(pos))
                    }
                },
            },
            AbstractInstruction::Effect {
                args,
                funcs,
                labels,
                op,
                #[cfg(feature = "position")]
                pos,
            } => Self::Effect {
                args,
                funcs,
                labels,
                #[cfg(feature = "position")]
                pos: pos.clone(),
                op: match op.as_ref() {
                    "jmp" => EffectOps::Jump,
                    "br" => EffectOps::Branch,
                    "call" => EffectOps::Call,
                    "ret" => EffectOps::Return,
                    "print" => EffectOps::Print,
                    "nop" => EffectOps::Nop,
                    #[cfg(feature = "memory")]
                    "store" => EffectOps::Store,
                    #[cfg(feature = "memory")]
                    "free" => EffectOps::Free,
                    #[cfg(feature = "speculate")]
                    "speculate" => EffectOps::Speculate,
                    #[cfg(feature = "speculate")]
                    "commit" => EffectOps::Commit,
                    #[cfg(feature = "speculate")]
                    "guard" => EffectOps::Guard,
                    e => {
                        return Err(ConversionError::InvalidEffectOps(e.to_string()))
                            .map_err(|e| e.add_pos(pos))
                    }
                },
            },
        })
    }
}

impl TryFrom<Option<AbstractType>> for Type {
    type Error = ConversionError;

    fn try_from(value: Option<AbstractType>) -> Result<Self, Self::Error> {
        value.map_or(Err(ConversionError::MissingType), TryInto::try_into)
    }
}

impl TryFrom<AbstractType> for Type {
    type Error = ConversionError;
    fn try_from(value: AbstractType) -> Result<Self, Self::Error> {
        Ok(match value {
            AbstractType::Primitive(t) if t == "int" => Self::Int,
            AbstractType::Primitive(t) if t == "bool" => Self::Bool,
            #[cfg(feature = "float")]
            AbstractType::Primitive(t) if t == "float" => Self::Float,
            #[cfg(feature = "char")]
            AbstractType::Primitive(t) if t == "char" => Self::Char,
            AbstractType::Primitive(t) => return Err(ConversionError::InvalidPrimitive(t)),
            #[cfg(feature = "memory")]
            AbstractType::Parameterized(t, ty) if t == "ptr" => {
                Self::Pointer(Box::new((*ty).try_into()?))
            }
            AbstractType::Parameterized(t, ty) => {
                return Err(ConversionError::InvalidParameterized(t, ty.to_string()))
            }
        })
    }
}
