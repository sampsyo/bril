use std::convert::TryFrom;
use std::fmt;

use crate::basic_block::BBProgram;
use crate::ir_types::{BrArgs, BrilType, BrilValue, EffectOp, Identifier, Operation, ValueOp};

impl BrilValue {
  fn get_type(&self) -> BrilType {
    match self {
      BrilValue::Int(..) => BrilType::Int,
      BrilValue::Bool(..) => BrilType::Bool,
      BrilValue::Nil => BrilType::Nil,
    }
  }
}

impl fmt::Display for BrilValue {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    use BrilValue::*;
    match self {
      Int(i) => write!(f, "{}", i),
      Bool(b) => write!(f, "{}", b),
      Nil => write!(f, "nil"),
    }
  }
}

impl TryFrom<&BrilValue> for i64 {
  type Error = InterpError;
  fn try_from(value: &BrilValue) -> Result<Self, Self::Error> {
    match value {
      BrilValue::Int(i) => Ok(*i),
      _ => Err(InterpError::BadValueType(BrilType::Int, value.get_type())),
    }
  }
}

impl TryFrom<&BrilValue> for bool {
  type Error = InterpError;
  fn try_from(value: &BrilValue) -> Result<Self, Self::Error> {
    match value {
      BrilValue::Bool(b) => Ok(*b),
      _ => Err(InterpError::BadValueType(BrilType::Bool, value.get_type())),
    }
  }
}

#[derive(Debug)]
pub enum InterpError {
  BadJsonInt,
  BadJsonBool,
  NoMainFunction,
  BadNumArgs(usize, usize), // (expected, actual)
  VarNotFound(usize),
  BadAsmtType(BrilType, BrilType), // (expected, actual). For when the LHS type of an instruction is bad
  LabelNotFound(String),
  BadValueType(BrilType, BrilType), // (expected, actual)
  IoError(Box<std::io::Error>),
}

fn check_asmt_type(expected: &BrilType, actual: &BrilType) -> Result<(), InterpError> {
  if expected == actual {
    Ok(())
  } else {
    Err(InterpError::BadAsmtType(expected.clone(), actual.clone()))
  }
}

fn get_args<'a, T>(
  vars: &'a Vec<BrilValue>,
  arity: usize,
  args: &Vec<Identifier<usize>>,
) -> Result<Vec<T>, InterpError>
where
  T: TryFrom<&'a BrilValue>,
  InterpError: std::convert::From<<T as TryFrom<&'a BrilValue>>::Error>,
  <T as TryFrom<&'a BrilValue>>::Error: std::convert::From<InterpError>,
{
  if args.len() != arity {
    return Err(InterpError::BadNumArgs(arity, args.len()));
  }

  let mut arg_vals = vec![];
  for arg in args {
    let arg_bril_val = vars
      .get(arg.0)
      // TODO: This error message will be pretty uninformative without a map back from Identifier ->
      // String
      .ok_or_else(|| InterpError::VarNotFound(arg.0))?;
    arg_vals.push(T::try_from(arg_bril_val)?);
  }

  Ok(arg_vals)
}

pub fn execute<T: std::io::Write>(
  prog: BBProgram,
  num_vars: usize,
  mut out: T,
) -> Result<(), InterpError> {
  let (main_fn, blocks, _labels) = prog;
  let mut curr_block_idx: usize = main_fn.ok_or(InterpError::NoMainFunction)?;
  let mut store: Vec<BrilValue> = vec![BrilValue::Nil; num_vars];

  use BrilValue::*;
  use Operation::*;

  loop {
    let curr_block = &blocks[curr_block_idx];
    let curr_instrs = &curr_block.instrs;
    let mut next_block_idx = if curr_block.exit.len() == 1 {
      Some(curr_block.exit[0])
    } else {
      None
    };

    for operation in curr_instrs.iter() {
      match operation {
        Const { dest, typ, value } => {
          check_asmt_type(typ, &value.get_type())?;
          store[dest.0] = value.clone();
        }
        Add {
          params: ValueOp { dest, typ, args },
        } => {
          check_asmt_type(&BrilType::Int, typ)?;
          let args = get_args::<i64>(&store, 2, args)?;
          store[dest.0] = Int(args[0] + args[1]);
        }
        Mul {
          params: ValueOp { dest, typ, args },
        } => {
          check_asmt_type(&BrilType::Int, typ)?;
          let args = get_args::<i64>(&store, 2, args)?;
          store[dest.0] = Int(args[0] * args[1]);
        }
        Sub {
          params: ValueOp { dest, typ, args },
        } => {
          check_asmt_type(&BrilType::Int, typ)?;
          let args = get_args::<i64>(&store, 2, args)?;
          store[dest.0] = Int(args[0] - args[1]);
        }
        Div {
          params: ValueOp { dest, typ, args },
        } => {
          check_asmt_type(&BrilType::Int, typ)?;
          let args = get_args::<i64>(&store, 2, args)?;
          store[dest.0] = Int(args[0] / args[1]);
        }
        Eq {
          params: ValueOp { dest, typ, args },
        } => {
          check_asmt_type(&BrilType::Bool, typ)?;
          let args = get_args::<i64>(&store, 2, args)?;
          store[dest.0] = Bool(args[0] == args[1]);
        }
        Lt {
          params: ValueOp { dest, typ, args },
        } => {
          check_asmt_type(&BrilType::Bool, typ)?;
          let args = get_args::<i64>(&store, 2, args)?;
          store[dest.0] = Bool(args[0] < args[1]);
        }
        Gt {
          params: ValueOp { dest, typ, args },
        } => {
          check_asmt_type(&BrilType::Bool, typ)?;
          let args = get_args::<i64>(&store, 2, args)?;
          store[dest.0] = Bool(args[0] > args[1]);
        }
        Le {
          params: ValueOp { dest, typ, args },
        } => {
          check_asmt_type(&BrilType::Bool, typ)?;
          let args = get_args::<i64>(&store, 2, args)?;
          store[dest.0] = Bool(args[0] <= args[1]);
        }
        Ge {
          params: ValueOp { dest, typ, args },
        } => {
          check_asmt_type(&BrilType::Bool, typ)?;
          let args = get_args::<i64>(&store, 2, args)?;
          store[dest.0] = Bool(args[0] >= args[1]);
        }
        Not {
          params: ValueOp { dest, typ, args },
        } => {
          check_asmt_type(&BrilType::Bool, typ)?;
          let args = get_args::<bool>(&store, 2, args)?;
          store[dest.0] = Bool(!args[0]);
        }
        And {
          params: ValueOp { dest, typ, args },
        } => {
          check_asmt_type(&BrilType::Bool, typ)?;
          let args = get_args::<bool>(&store, 2, args)?;
          store[dest.0] = Bool(args[0] && args[1]);
        }
        Or {
          params: ValueOp { dest, typ, args },
        } => {
          check_asmt_type(&BrilType::Bool, typ)?;
          let args = get_args::<bool>(&store, 2, args)?;
          store[dest.0] = Bool(args[0] || args[1]);
        }
        Jmp {
          params: EffectOp { args },
        } => {
          if args.len() != 1 {
            return Err(InterpError::BadNumArgs(1, args.len()));
          }

          next_block_idx = Some(curr_block.exit[0]);
        }
        Br {
          params: BrArgs::IdArgs { test_var, dests },
        } => {
          if dests.len() != 2 {
            return Err(InterpError::BadNumArgs(3, dests.len()));
          }

          let exit_idx = if bool::try_from(&store[test_var.0])? {
            0
          } else {
            1
          };

          next_block_idx = Some(curr_block.exit[exit_idx]);
        }
        Ret { .. } => {
          out.flush().map_err(|e| InterpError::IoError(Box::new(e)))?;
          // NOTE: This only works so long as `main` is the only function
          return Ok(());
        }
        Id {
          params: ValueOp { dest, typ, args },
        } => {
          if args.len() != 1 {
            return Err(InterpError::BadNumArgs(1, args.len()));
          }

          let src = store[args[0].0].clone();
          check_asmt_type(&src.get_type(), typ)?;
          store[dest.0] = src;
        }
        Print { args } => {
          write!(
            out,
            // NOTE: The Bril spec implies print should just output its arguments, with no newline.
            // However, brili uses console.log, which does add a newline, so we will too
            "{}\n",
            args
              .iter()
              .map(|a| format!("{}", store[a.0]))
              .collect::<Vec<_>>()
              .join(", ")
          )
          .map_err(|e| InterpError::IoError(Box::new(e)))?;
        }
        Nop => {}
        _ => unreachable!(),
      }
    }

    if let Some(idx) = next_block_idx {
      curr_block_idx = idx;
    } else {
      out.flush().map_err(|e| InterpError::IoError(Box::new(e)))?;
      return Ok(());
    }
  }
}
