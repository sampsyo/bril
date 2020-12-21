use std::collections::HashMap;
use std::convert::TryFrom;
use std::fmt;

use crate::basic_block::{BBProgram, BasicBlock};

#[derive(Debug)]
pub enum InterpError {
  BadJsonInt,
  BadJsonBool,
  NoMainFunction,
  BadNumArgs(usize, usize),   // (expected, actual)
  BadNumLabels(usize, usize), // (expected, actual)
  VarNotFound(String),
  BadAsmtType(bril_rs::Type, bril_rs::Type), // (expected, actual). For when the LHS type of an instruction is bad
  LabelNotFound(String),
  BadValueType(bril_rs::Type, bril_rs::Type), // (expected, actual)
  IoError(Box<std::io::Error>),
}

fn check_asmt_type(expected: &bril_rs::Type, actual: &bril_rs::Type) -> Result<(), InterpError> {
  if expected == actual {
    Ok(())
  } else {
    Err(InterpError::BadAsmtType(expected.clone(), actual.clone()))
  }
}

fn get_values<'a>(
  vars: &'a HashMap<String, Value>,
  arity: usize,
  args: &[String],
) -> Result<Vec<&'a Value>, InterpError> {
  if args.len() != arity {
    return Err(InterpError::BadNumArgs(arity, args.len()));
  }

  let mut vals = vec![];
  for arg in args {
    let arg_bril_val = vars
      .get(arg)
      .ok_or_else(|| InterpError::VarNotFound(arg.clone()))?;
    vals.push(arg_bril_val);
  }

  Ok(vals)
}

fn get_args<'a, T>(
  vars: &'a HashMap<String, Value>,
  arity: usize,
  args: &[String],
) -> Result<Vec<T>, InterpError>
where
  T: TryFrom<&'a Value>,
  InterpError: std::convert::From<<T as TryFrom<&'a Value>>::Error>,
  <T as TryFrom<&'a Value>>::Error: std::convert::From<InterpError>,
{
  if args.len() != arity {
    return Err(InterpError::BadNumArgs(arity, args.len()));
  }

  let mut arg_vals = vec![];
  for arg in args {
    let arg_bril_val = vars
      .get(arg)
      .ok_or_else(|| InterpError::VarNotFound(arg.clone()))?;
    arg_vals.push(T::try_from(arg_bril_val)?);
  }

  Ok(arg_vals)
}

#[derive(Debug, Clone)]
enum Value {
  Int(i64),
  Bool(bool),
  Float(f64),
}

impl Value {
  pub fn get_type(&self) -> bril_rs::Type {
    match *self {
      Value::Int(_) => bril_rs::Type::Int,
      Value::Bool(_) => bril_rs::Type::Bool,
      Value::Float(_) => bril_rs::Type::Float,
    }
  }
}

impl fmt::Display for Value {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    use Value::*;
    match self {
      Int(i) => write!(f, "{}", i),
      Bool(b) => write!(f, "{}", b),
      Float(v) => write!(f, "{}", v),
    }
  }
}

impl From<&bril_rs::Literal> for Value {
  fn from(l: &bril_rs::Literal) -> Value {
    match l {
      bril_rs::Literal::Int(i) => Value::Int(*i),
      bril_rs::Literal::Bool(b) => Value::Bool(*b),
      bril_rs::Literal::Float(f) => Value::Float(*f),
    }
  }
}

impl TryFrom<&Value> for i64 {
  type Error = InterpError;
  fn try_from(value: &Value) -> Result<Self, Self::Error> {
    if let Value::Int(i) = value {
      Ok(*i)
    } else {
      Err(InterpError::BadValueType(
        bril_rs::Type::Int,
        value.get_type(),
      ))
    }
  }
}

impl TryFrom<&Value> for bool {
  type Error = InterpError;
  fn try_from(value: &Value) -> Result<Self, Self::Error> {
    if let Value::Bool(b) = value {
      Ok(*b)
    } else {
      Err(InterpError::BadValueType(
        bril_rs::Type::Bool,
        value.get_type(),
      ))
    }
  }
}

impl TryFrom<&Value> for f64 {
  type Error = InterpError;
  fn try_from(value: &Value) -> Result<Self, Self::Error> {
    match value {
      Value::Float(f) => Ok(*f),
      _ => Err(InterpError::BadValueType(
        bril_rs::Type::Float,
        value.get_type(),
      )),
    }
  }
}

#[allow(clippy::float_cmp)]
fn execute_value_op(
  op: &bril_rs::ValueOps,
  dest: &str,
  op_type: &bril_rs::Type,
  args: &[String],
  value_store: &mut HashMap<String, Value>,
) -> Result<(), InterpError> {
  use bril_rs::ValueOps::*;
  match *op {
    Add => {
      check_asmt_type(&bril_rs::Type::Int, op_type)?;
      let args = get_args::<i64>(value_store, 2, args)?;
      value_store.insert(String::from(dest), Value::Int(args[0] + args[1]));
    }
    Mul => {
      check_asmt_type(&bril_rs::Type::Int, op_type)?;
      let args = get_args::<i64>(value_store, 2, args)?;
      value_store.insert(String::from(dest), Value::Int(args[0] * args[1]));
    }
    Sub => {
      check_asmt_type(&bril_rs::Type::Int, op_type)?;
      let args = get_args::<i64>(value_store, 2, args)?;
      value_store.insert(String::from(dest), Value::Int(args[0] - args[1]));
    }
    Div => {
      check_asmt_type(&bril_rs::Type::Int, op_type)?;
      let args = get_args::<i64>(value_store, 2, args)?;
      value_store.insert(String::from(dest), Value::Int(args[0] / args[1]));
    }
    Eq => {
      check_asmt_type(&bril_rs::Type::Bool, op_type)?;
      let args = get_args::<i64>(value_store, 2, args)?;
      value_store.insert(String::from(dest), Value::Bool(args[0] == args[1]));
    }
    Lt => {
      check_asmt_type(&bril_rs::Type::Bool, op_type)?;
      let args = get_args::<i64>(value_store, 2, args)?;
      value_store.insert(String::from(dest), Value::Bool(args[0] < args[1]));
    }
    Gt => {
      check_asmt_type(&bril_rs::Type::Bool, op_type)?;
      let args = get_args::<i64>(value_store, 2, args)?;
      value_store.insert(String::from(dest), Value::Bool(args[0] > args[1]));
    }
    Le => {
      check_asmt_type(&bril_rs::Type::Bool, op_type)?;
      let args = get_args::<i64>(value_store, 2, args)?;
      value_store.insert(String::from(dest), Value::Bool(args[0] <= args[1]));
    }
    Ge => {
      check_asmt_type(&bril_rs::Type::Bool, op_type)?;
      let args = get_args::<i64>(value_store, 2, args)?;
      value_store.insert(String::from(dest), Value::Bool(args[0] >= args[1]));
    }
    Not => {
      check_asmt_type(&bril_rs::Type::Bool, op_type)?;
      let args = get_args::<bool>(value_store, 1, args)?;
      value_store.insert(String::from(dest), Value::Bool(!args[0]));
    }
    And => {
      check_asmt_type(&bril_rs::Type::Bool, op_type)?;
      let args = get_args::<bool>(value_store, 2, args)?;
      value_store.insert(String::from(dest), Value::Bool(args[0] && args[1]));
    }
    Or => {
      check_asmt_type(&bril_rs::Type::Bool, op_type)?;
      let args = get_args::<bool>(value_store, 2, args)?;
      value_store.insert(String::from(dest), Value::Bool(args[0] || args[1]));
    }
    Id => {
      check_num_args(1, args)?;
      let src_vals = get_values(value_store, 1, args)?;
      let src = src_vals[0].clone();
      value_store.insert(String::from(dest), src);
    }
    Fadd => {
      check_asmt_type(&bril_rs::Type::Float, op_type)?;
      let args = get_args::<f64>(value_store, 2, args)?;
      value_store.insert(String::from(dest), Value::Float(args[0] + args[1]));
    }
    Fmul => {
      check_asmt_type(&bril_rs::Type::Float, op_type)?;
      let args = get_args::<f64>(value_store, 2, args)?;
      value_store.insert(String::from(dest), Value::Float(args[0] * args[1]));
    }
    Fsub => {
      check_asmt_type(&bril_rs::Type::Float, op_type)?;
      let args = get_args::<f64>(value_store, 2, args)?;
      value_store.insert(String::from(dest), Value::Float(args[0] - args[1]));
    }
    Fdiv => {
      check_asmt_type(&bril_rs::Type::Float, op_type)?;
      let args = get_args::<f64>(value_store, 2, args)?;
      value_store.insert(String::from(dest), Value::Float(args[0] / args[1]));
    }
    Feq => {
      check_asmt_type(&bril_rs::Type::Float, op_type)?;
      let args = get_args::<f64>(value_store, 2, args)?;
      value_store.insert(String::from(dest), Value::Bool(args[0] == args[1]));
    }
    Flt => {
      check_asmt_type(&bril_rs::Type::Float, op_type)?;
      let args = get_args::<f64>(value_store, 2, args)?;
      value_store.insert(String::from(dest), Value::Bool(args[0] < args[1]));
    }
    Fgt => {
      check_asmt_type(&bril_rs::Type::Float, op_type)?;
      let args = get_args::<f64>(value_store, 2, args)?;
      value_store.insert(String::from(dest), Value::Bool(args[0] > args[1]));
    }
    Fle => {
      check_asmt_type(&bril_rs::Type::Float, op_type)?;
      let args = get_args::<f64>(value_store, 2, args)?;
      value_store.insert(String::from(dest), Value::Bool(args[0] <= args[1]));
    }
    Fge => {
      check_asmt_type(&bril_rs::Type::Float, op_type)?;
      let args = get_args::<f64>(value_store, 2, args)?;
      value_store.insert(String::from(dest), Value::Bool(args[0] >= args[1]));
    }
    Call => unreachable!(), // TODO(yati): Why is Call a ValueOp as well?
    Phi | Alloc | Load | PtrAdd => unimplemented!(),
  }
  Ok(())
}

fn check_num_args(expected: usize, args: &[String]) -> Result<(), InterpError> {
  if expected != args.len() {
    Err(InterpError::BadNumArgs(expected, args.len()))
  } else {
    Ok(())
  }
}

fn check_num_labels(expected: usize, labels: &[String]) -> Result<(), InterpError> {
  if expected != labels.len() {
    Err(InterpError::BadNumArgs(expected, labels.len()))
  } else {
    Ok(())
  }
}

// Returns whether the program should continue running (i.e., if a Return was
// *not* executed).
fn execute_effect_op<T: std::io::Write>(
  op: &bril_rs::EffectOps,
  args: &[String],
  labels: &[String],
  curr_block: &BasicBlock,
  value_store: &HashMap<String, Value>,
  mut out: T,
  next_block_idx: &mut Option<usize>,
) -> Result<bool, InterpError> {
  use bril_rs::EffectOps::*;
  match op {
    Jump => {
      check_num_args(0, args)?;
      check_num_labels(1, labels)?;
      *next_block_idx = Some(curr_block.exit[0]);
    }
    Branch => {
      let bool_args = get_args::<bool>(value_store, 1, args)?;
      check_num_labels(2, labels)?;
      let exit_idx = if bool_args[0] { 0 } else { 1 };
      *next_block_idx = Some(curr_block.exit[exit_idx]);
    }
    Return => {
      out.flush().map_err(|e| InterpError::IoError(Box::new(e)))?;
      // NOTE: This only works so long as `main` is the only function
      return Ok(false);
    }
    Print => {
      writeln!(
        out,
        "{}",
        args
          .iter()
          .map(|a| format!("{}", value_store[a]))
          .collect::<Vec<_>>()
          .join(", ")
      )
      .map_err(|e| InterpError::IoError(Box::new(e)))?;
    }
    Nop => {}
    Call => unreachable!(),
    Store | Free | Speculate | Commit | Guard => unimplemented!(),
  }
  Ok(true)
}

pub fn execute<T: std::io::Write>(prog: BBProgram, mut out: T) -> Result<(), InterpError> {
  let (main_fn, blocks, _labels) = prog;
  let mut curr_block_idx: usize = main_fn.ok_or(InterpError::NoMainFunction)?;

  // Map from variable name to value.
  let mut value_store: HashMap<String, Value> = HashMap::new();

  loop {
    let curr_block = &blocks[curr_block_idx];
    let curr_instrs = &curr_block.instrs;
    let mut next_block_idx = if curr_block.exit.len() == 1 {
      Some(curr_block.exit[0])
    } else {
      None
    };

    for operation in curr_instrs {
      if let bril_rs::Code::Instruction(instr) = operation {
        match instr {
          bril_rs::Instruction::Constant {
            op: bril_rs::ConstOps::Const,
            dest,
            const_type,
            value,
          } => {
            check_asmt_type(const_type, &value.get_type())?;
            value_store.insert(dest.clone(), Value::from(value));
          }
          bril_rs::Instruction::Value {
            op,
            dest,
            op_type,
            args,
            ..
          } => {
            execute_value_op(op, dest, op_type, args, &mut value_store)?;
          }
          bril_rs::Instruction::Effect {
            op, args, labels, ..
          } => {
            let should_continue = execute_effect_op(
              op,
              args,
              labels,
              &curr_block,
              &value_store,
              &mut out,
              &mut next_block_idx,
            )?;

            // TODO(yati): Correct only when main is the only function.
            if !should_continue {
              return Ok(());
            }
          }
        }
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
