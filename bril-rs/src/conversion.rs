use crate::{
    AbstractArgument, AbstractCode, AbstractFunction, AbstractInstruction, AbstractProgram,
    AbstractType, Argument, Code, EffectOps, Function, Instruction, Program, Type, ValueOps,
};

use thiserror::Error;

// Having the #[error(...)] for all variants derives the Display trait as well
#[derive(Error, Debug)]
#[allow(clippy::module_name_repetitions)]
pub enum ConversionError {
    #[error("Expected a primitive type like int or bool, found {0}")]
    InvalidPrimitive(String),

    #[error("Expected a parameterized type like ptr, found {0}<{1}>")]
    InvalidParameterized(String, String),

    #[error("Expected an value operation, found {0}")]
    InvalidValueOps(String),

    #[error("Expected an effect operation, found {0}")]
    InvalidEffectOps(String),
}

impl TryFrom<AbstractProgram> for Program {
    type Error = ConversionError;
    fn try_from(AbstractProgram { functions }: AbstractProgram) -> Result<Self, Self::Error> {
        Ok(Self {
            functions: functions
                .into_iter()
                .map(std::convert::TryInto::try_into)
                .collect::<Result<Vec<Function>, _>>()?,
        })
    }
}

impl TryFrom<AbstractFunction> for Function {
    type Error = ConversionError;
    fn try_from(
        AbstractFunction {
            args,
            instrs,
            name,
            return_type,
        }: AbstractFunction,
    ) -> Result<Self, Self::Error> {
        Ok(Self {
            args: args
                .into_iter()
                .map(std::convert::TryInto::try_into)
                .collect::<Result<Vec<Argument>, _>>()?,
            instrs: instrs
                .into_iter()
                .map(std::convert::TryInto::try_into)
                .collect::<Result<Vec<Code>, _>>()?,
            name,
            return_type: match return_type {
                None => None,
                Some(t) => Some(t.try_into()?),
            },
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
    type Error = ConversionError;
    fn try_from(c: AbstractCode) -> Result<Self, Self::Error> {
        Ok(match c {
            AbstractCode::Label { label } => Self::Label { label },
            AbstractCode::Instruction(i) => Self::Instruction(i.try_into()?),
        })
    }
}

impl TryFrom<AbstractInstruction> for Instruction {
    type Error = ConversionError;
    fn try_from(i: AbstractInstruction) -> Result<Self, Self::Error> {
        Ok(match i {
            AbstractInstruction::Constant {
                dest,
                op,
                const_type,
                value,
            } => Self::Constant {
                dest,
                op,
                const_type: const_type.try_into()?,
                value,
            },
            AbstractInstruction::Value {
                args,
                dest,
                funcs,
                labels,
                op,
                op_type,
            } => Self::Value {
                args,
                dest,
                funcs,
                labels,
                op_type: op_type.try_into()?,
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
                    #[cfg(feature = "memory")]
                    "alloc" => ValueOps::Alloc,
                    #[cfg(feature = "memory")]
                    "load" => ValueOps::Load,
                    #[cfg(feature = "memory")]
                    "ptradd" => ValueOps::PtrAdd,
                    v => return Err(ConversionError::InvalidValueOps(v.to_string())),
                },
            },
            AbstractInstruction::Effect {
                args,
                funcs,
                labels,
                op,
            } => Self::Effect {
                args,
                funcs,
                labels,
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
                    e => return Err(ConversionError::InvalidEffectOps(e.to_string())),
                },
            },
        })
    }
}

impl TryFrom<AbstractType> for Type {
    type Error = ConversionError;
    fn try_from(value: AbstractType) -> Result<Self, Self::Error> {
        Ok(match value {
            AbstractType::Primitive(t) if t == "int" => Self::Int,
            AbstractType::Primitive(t) if t == "bool" => Self::Bool,
            #[cfg(feature = "float")]
            AbstractType::Primitive(t) if t == "float" => Type::Float,
            AbstractType::Primitive(t) => return Err(ConversionError::InvalidPrimitive(t)),
            #[cfg(feature = "memory")]
            AbstractType::Parameterized(t, ty) if t == "ptr" => {
                Type::Pointer(Box::new((*ty).try_into()?))
            }
            AbstractType::Parameterized(t, ty) => {
                return Err(ConversionError::InvalidParameterized(t, ty.to_string()))
            }
        })
    }
}
