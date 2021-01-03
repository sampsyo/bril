use std::collections::HashMap;
use std::convert::TryFrom;
use std::fmt;

use crate::basic_block::{BBProgram, BasicBlock, Function};

#[derive(Debug)]
pub enum InterpError {
  BadJsonInt,
  BadJsonBool,
  NoMainFunction,
  FuncNotFound(String),
  NoRetValForfunc(String),
  BadNumArgs(usize, usize),   // (expected, actual)
  BadNumLabels(usize, usize), // (expected, actual)
  VarNotFound(String),
  BadAsmtType(bril_rs::Type, bril_rs::Type), // (expected, actual). For when the LHS type of an instruction is bad
  LabelNotFound(String),
  BadValueType(bril_rs::Type, bril_rs::Type), // (expected, actual)
  IoError(Box<std::io::Error>),
  BadCall(String, String), // (func name, reason).
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
fn execute_value_op<W: std::io::Write>(
  prog: &BBProgram,
  op: &bril_rs::ValueOps,
  dest: &str,
  op_type: &bril_rs::Type,
  args: &[String],
  funcs: &[String],
  value_store: &mut HashMap<String, Value>,
  out: &mut W,
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
    Call => {
      assert!(funcs.len() == 1);
      let func_info = prog
        .func_index
        .get(&funcs[0])
        .ok_or(InterpError::FuncNotFound(funcs[0].clone()))?;

      check_asmt_type(
        func_info.return_type.as_ref().ok_or(InterpError::BadCall(
          String::from(&funcs[0]),
          String::from(
            "Function does not return a value, but used on the right side of an assignment",
          ),
        ))?,
        op_type,
      )?;

      let vars = make_func_args(&funcs[0], func_info, args, value_store)?;
      if let Some(val) = execute_func(&prog, &funcs[0], vars, out)? {
        check_asmt_type(&val.get_type(), op_type)?;
        value_store.insert(String::from(dest), val);
      } else {
        // This is a value-op call, so the target func must return a result.
        return Err(InterpError::NoRetValForfunc(funcs[0].clone()));
      }
    }
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

// Returns a map from function parameter names to values of the call arguments
// that are bound to those parameters.
fn make_func_args(
  func_name: &str,
  func: &Function,
  call_args: &[String],
  vars: &HashMap<String, Value>,
) -> Result<HashMap<String, Value>, InterpError> {
  if func.args.len() != call_args.len() {
    return Err(InterpError::BadCall(
      String::from(func_name),
      format!(
        "Expected {} parameters, tried to pass {} args",
        func.args.len(),
        call_args.len()
      ),
    ));
  }
  let vals = get_values(vars, call_args.len(), call_args)?;
  let mut args = HashMap::new();
  for (i, arg) in func.args.iter().enumerate() {
    check_asmt_type(&arg.arg_type, &vals[i].get_type())?;
    args.insert(arg.name.clone(), vals[i].clone());
  }
  Ok(args)
}

// Result of executing an effect operation.
enum EffectResult {
  // Return from the current function without any value.
  Return,

  // Return a given value from the current function.
  ReturnWithVal(Value),

  // Continue execution of the current function.
  Continue,
}

fn execute_effect_op<T: std::io::Write>(
  prog: &BBProgram,
  op: &bril_rs::EffectOps,
  args: &[String],
  labels: &[String],
  funcs: &[String],
  curr_block: &BasicBlock,
  value_store: &HashMap<String, Value>,
  out: &mut T,
  next_block_idx: &mut Option<usize>,
) -> Result<EffectResult, InterpError> {
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
      if args.is_empty() {
        return Ok(EffectResult::Return);
      }
      let retval = value_store
        .get(&args[0])
        .ok_or(InterpError::VarNotFound(args[0].clone()))?;
      return Ok(EffectResult::ReturnWithVal(retval.clone()));
    }
    Print => {
      let vals = get_values(value_store, args.len(), args)?;
      writeln!(
        out,
        "{}",
        vals
          .iter()
          .map(|v| format!("{}", v))
          .collect::<Vec<_>>()
          .join(", ")
      )
      .map_err(|e| InterpError::IoError(Box::new(e)))?;
    }
    Nop => {}
    Call => {
      assert!(funcs.len() == 1);
      let func = prog
        .func_index
        .get(&funcs[0])
        .ok_or(InterpError::FuncNotFound(funcs[0].clone()))?;
      let vars = make_func_args(&funcs[0], func, args, value_store)?;
      execute_func(&prog, &funcs[0], vars, out)?;
    }
    Store | Free | Speculate | Commit | Guard => unimplemented!(),
  }
  Ok(EffectResult::Continue)
}

fn execute_func<T: std::io::Write>(
  prog: &BBProgram,
  func: &str,
  mut vars: HashMap<String, Value>,
  out: &mut T,
) -> Result<Option<Value>, InterpError> {
  let f = prog
    .func_index
    .get(func)
    .ok_or(InterpError::FuncNotFound(String::from(func)))?;
  let mut curr_block_idx = 0;
  loop {
    let curr_block = &f.blocks[curr_block_idx];
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
            vars.insert(dest.clone(), Value::from(value));
          }
          bril_rs::Instruction::Value {
            op,
            dest,
            op_type,
            args,
            funcs,
            ..
          } => {
            execute_value_op(&prog, op, dest, op_type, args, funcs, &mut vars, out)?;
          }
          bril_rs::Instruction::Effect {
            op,
            args,
            labels,
            funcs,
            ..
          } => {
            match execute_effect_op(
              prog,
              op,
              args,
              labels,
              funcs,
              &curr_block,
              &vars,
              out,
              &mut next_block_idx,
            )? {
              EffectResult::Continue => {}
              EffectResult::Return => {
                return Ok(None);
              }
              EffectResult::ReturnWithVal(val) => {
                return Ok(Some(val));
              }
            };
          }
        }
      }
    }
    if let Some(idx) = next_block_idx {
      curr_block_idx = idx;
    } else {
      out.flush().map_err(|e| InterpError::IoError(Box::new(e)))?;
      return Ok(None);
    }
  }
}

pub fn execute<T: std::io::Write>(prog: BBProgram, out: &mut T) -> Result<(), InterpError> {
  // Ignore return value of @main.
  execute_func(&prog, "main", HashMap::new(), out).map(|_| ())
}
