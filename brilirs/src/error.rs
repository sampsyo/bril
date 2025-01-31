use std::fmt::Display;

use bril_rs::{conversion::PositionalConversionError, Position};
use std::error::Error;
use thiserror::Error;

// Having the #[error(...)] for all variants derives the Display trait as well
#[derive(Error, Debug)]
pub enum InterpError {
  #[error("Attempt to divide by 0")]
  DivisionByZero,
  #[error("Some memory locations have not been freed by the end of execution")]
  MemLeak,
  #[error("Trying to load from uninitialized memory")]
  UsingUninitializedMemory,
  #[error("phi node executed with no last label")]
  NoLastLabel,
  #[error("Could not find label: {0}")]
  MissingLabel(String),
  #[error("no main function defined, doing nothing")]
  NoMainFunction,
  #[error("phi node has unequal numbers of labels and args")]
  UnequalPhiNode,
  #[error("char must have one character")]
  NotOneChar,
  #[error("multiple functions of the same name found")]
  DuplicateFunction,
  #[error("duplicate label `{0}` found")]
  DuplicateLabel(String),
  #[error("Expected empty return for `{0}`, found value")]
  NonEmptyRetForFunc(String),
  #[error("cannot allocate `{0}` entries")]
  CannotAllocSize(i64),
  #[error("Tried to free illegal memory location base: `{0}`, offset: `{1}`. Offset must be 0.")]
  IllegalFree(usize, i64), // (base, offset)
  #[error("Uninitialized heap location `{0}` and/or illegal offset `{1}`")]
  InvalidMemoryAccess(usize, i64), // (base, offset)
  #[error("Expected `{0}` function arguments, found `{1}`")]
  BadNumFuncArgs(usize, usize), // (expected, actual)
  #[error("Expected `{0}` instruction arguments, found `{1}`")]
  BadNumArgs(usize, usize), // (expected, actual)
  #[error("Expected `{0}` labels, found `{1}`")]
  BadNumLabels(usize, usize), // (expected, actual)
  #[error("Expected `{0}` functions, found `{1}`")]
  BadNumFuncs(usize, usize), // (expected, actual)
  #[error("no function of name `{0}` found")]
  FuncNotFound(String),
  #[error("undefined variable `{0}`")]
  VarUndefined(String),
  #[error("Label `{0}` for phi node not found")]
  PhiMissingLabel(String),
  #[error("unspecified pointer type `{0:?}`")]
  ExpectedPointerType(bril_rs::Type), // found type
  #[error("Expected type `{0:?}` for function argument, found `{1:?}`")]
  BadFuncArgType(bril_rs::Type, String), // (expected, actual)
  #[error("Expected type `{0:?}` for assignment, found `{1:?}`")]
  BadAsmtType(bril_rs::Type, bril_rs::Type), // (expected, actual). For when the LHS type of an instruction is bad
  #[error("There has been an io error: `{0:?}`")]
  IoError(#[from] std::io::Error),
  #[error("value ${0} cannot be converted to char")]
  ToCharError(i64),
  #[error("You probably shouldn't see this error, this is here to handle conversions between InterpError and PositionalError")]
  PositionalInterpErrorConversion(#[from] PositionalInterpError),
}

impl InterpError {
  #[must_use]
  pub fn add_pos(self, pos: Option<Position>) -> PositionalInterpError {
    match self {
      Self::PositionalInterpErrorConversion(e) => e,
      _ => PositionalInterpError {
        e: Box::new(self),
        pos,
      },
    }
  }
}

#[derive(Error, Debug)]
pub struct PositionalInterpError {
  pub e: Box<dyn Error>,
  pub pos: Option<Position>,
}

impl Display for PositionalInterpError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self {
        e,
        pos:
          Some(Position {
            pos,
            pos_end: Some(end),
            src: Some(s),
          }),
      } => {
        write!(
          f,
          "{s}:{}:{} to {s}:{}:{} \n\t {e}",
          pos.row, pos.col, end.row, end.col
        )
      }
      Self {
        e,
        pos:
          Some(Position {
            pos,
            pos_end: None,
            src: Some(s),
          }),
      } => {
        write!(f, "{s}:{}:{} \n\t {e}", pos.row, pos.col)
      }
      Self {
        e,
        pos:
          Some(Position {
            pos,
            pos_end: Some(end),
            src: None,
          }),
      } => {
        write!(
          f,
          "Line {}, Column {} to Line {}, Column {}: {e}",
          pos.row, pos.col, end.row, end.col
        )
      }
      Self {
        e,
        pos: Some(Position {
          pos,
          pos_end: None,
          src: None,
        }),
      } => {
        write!(f, "Line {}, Column {}: {e}", pos.row, pos.col)
      }
      Self { e, pos: None } => write!(f, "{e}"),
    }
  }
}

impl From<InterpError> for PositionalInterpError {
  fn from(e: InterpError) -> Self {
    match e {
      InterpError::PositionalInterpErrorConversion(positional_e) => positional_e,
      _ => Self {
        e: Box::new(e),
        pos: None,
      },
    }
  }
}

impl From<PositionalConversionError> for PositionalInterpError {
  fn from(PositionalConversionError { e, pos }: PositionalConversionError) -> Self {
    Self {
      e: Box::new(e),
      pos,
    }
  }
}
