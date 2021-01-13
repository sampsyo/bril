#[derive(Debug)]
pub enum InterpError {
  MemLeak,
  UsingUninitializedMemory,
  NoLastLabel,
  NoMainFunction,
  UnequalPhiNode, // Unequal number of args and labels
  EmptyRetForfunc(String),
  NonEmptyRetForfunc(String),
  CannotAllocSize(i64),
  IllegalFree(usize, i64),         // (base, offset)
  InvalidMemoryAccess(usize, i64), // (base, offset)
  BadNumFuncArgs(usize, usize),    // (expected, actual)
  BadNumArgs(usize, usize),        // (expected, actual)
  BadNumLabels(usize, usize),      // (expected, actual)
  BadNumFuncs(usize, usize),       // (expected, actual)
  FuncNotFound(String),
  VarNotFound(String),
  PhiMissingLabel(String),
  ExpectedPointerType(bril_rs::Type),        // found type
  BadFuncArgType(bril_rs::Type, String),     // (expected, actual)
  BadAsmtType(bril_rs::Type, bril_rs::Type), // (expected, actual). For when the LHS type of an instruction is bad
  IoError(Box<std::io::Error>),
}
