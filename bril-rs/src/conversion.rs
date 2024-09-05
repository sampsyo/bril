use std::fmt::Display;

use crate::{
    AbstractArgument, AbstractCode, AbstractFunction, AbstractInstruction, AbstractProgram,
    AbstractType, Argument, Code, Function, Instruction, Position, Program, Type,
};

use thiserror::Error;

#[cfg(not(feature = "position"))]
#[expect(
    non_upper_case_globals,
    reason = "This is a nifty trick to supply a global value for pos when it is not defined"
)]
const pos: Option<Position> = None;

/// This is the [`std::error::Error`] implementation for `bril_rs`. This crate currently only supports errors from converting between [`AbstractProgram`] and [Program]
// todo Should this also wrap Serde errors? In this case, maybe change the name from ConversionError
// Having the #[error(...)] for all variants derives the Display trait as well
#[derive(Error, Debug)]
#[expect(
    clippy::module_name_repetitions,
    reason = "I allow the `Error` suffix for enums"
)]
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
                op: op.parse().map_err(|e: ConversionError| e.add_pos(pos))?,
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
                op: op.parse().map_err(|e: ConversionError| e.add_pos(pos))?,
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
        match value {
            AbstractType::Primitive(t) => t.parse(),
            #[cfg(feature = "memory")]
            AbstractType::Parameterized(t, ty) if t == "ptr" => {
                Ok(Self::Pointer(Box::new((*ty).try_into()?)))
            }
            AbstractType::Parameterized(t, ty) => {
                Err(ConversionError::InvalidParameterized(t, ty.to_string()))
            }
        }
    }
}
