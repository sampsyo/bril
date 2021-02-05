use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum InterpError {
  MemLeak,
  UsingUninitializedMemory,
  NoLastLabel,
  NoMainFunction,
  UnequalPhiNode, // Unequal number of args and labels
  DuplicateFunction,
  NonEmptyRetForfunc(String),
  CannotAllocSize(i64),
  IllegalFree(usize, i64),         // (base, offset)
  InvalidMemoryAccess(usize, i64), // (base, offset)
  BadNumFuncArgs(usize, usize),    // (expected, actual)
  BadNumArgs(usize, usize),        // (expected, actual)
  BadNumLabels(usize, usize),      // (expected, actual)
  BadNumFuncs(usize, usize),       // (expected, actual)
  FuncNotFound(String),
  VarUndefined(String),
  PhiMissingLabel(String),
  ExpectedPointerType(bril_rs::Type),        // found type
  BadFuncArgType(bril_rs::Type, String),     // (expected, actual)
  BadAsmtType(bril_rs::Type, bril_rs::Type), // (expected, actual). For when the LHS type of an instruction is bad
  IoError(Box<std::io::Error>),
}

impl Display for InterpError {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    match self {
      InterpError::MemLeak => {
        write!(
          f,
          "error: Some memory locations have not been freed by the end of execution"
        )
      }
      InterpError::UsingUninitializedMemory => {
        write!(f, "error: Trying to load from uninitialized memory")
      }
      InterpError::NoLastLabel => {
        write!(f, "error: phi node executed with no last label")
      }
      InterpError::NoMainFunction => {
        write!(f, "error: no main function defined, doing nothing")
      }
      InterpError::UnequalPhiNode => {
        write!(f, "error: pi node has unequal numbers of labels and args")
      }
      InterpError::DuplicateFunction => {
        write!(f, "error: multiple functions of the same name found")
      }
      InterpError::NonEmptyRetForfunc(func) => {
        write!(f, "error: Expected empty return for {}, found value", func)
      }
      InterpError::CannotAllocSize(amt) => {
        write!(f, "error: cannot allocate {} entries", amt)
      }
      InterpError::IllegalFree(base, offset) => {
        write!(
          f,
          "error: Tried to free illegal memory location base: {}, offset: {}. Offset must be 0.",
          base, offset
        )
      }
      InterpError::InvalidMemoryAccess(base, offset) => {
        write!(
          f,
          "error: Uninitialized heap location {} and/or illegal offset {}",
          base, offset
        )
      }
      InterpError::BadNumFuncArgs(expected, actual) => {
        write!(
          f,
          "error: Expected {} function arguments, found {}",
          expected, actual
        )
      }
      InterpError::BadNumArgs(expected, actual) => {
        write!(
          f,
          "error: Expected {} instruction arguments, found {}",
          expected, actual
        )
      }
      InterpError::BadNumLabels(expected, actual) => {
        write!(f, "error: Expected {} labels, found {}", expected, actual)
      }
      InterpError::BadNumFuncs(expected, actual) => {
        write!(
          f,
          "error: Expected {} functions, found {}",
          expected, actual
        )
      }
      InterpError::FuncNotFound(func) => {
        write!(f, "error: no function of name {} found", func)
      }
      InterpError::VarUndefined(v) => {
        write!(f, "error: undefined variable {}", v)
      }
      InterpError::PhiMissingLabel(label) => {
        write!(f, "error: Label {} for phi node not found", label)
      }
      InterpError::ExpectedPointerType(typ) => {
        write!(f, "error: unspecified pointer type {:?}", typ)
      }
      InterpError::BadFuncArgType(expected, actual) => {
        write!(
          f,
          "error: Expected type {:?} for function argument, found {:?}",
          expected, actual
        )
      }
      InterpError::BadAsmtType(expected, actual) => {
        write!(
          f,
          "error: Expected type {:?} for assignment, found {:?}",
          expected, actual
        )
      }
      InterpError::IoError(err) => {
        write!(
          f,
          "error: There has been an io error when trying to print: {:?}",
          err
        )
      }
    }
  }
}
