use std::convert::TryFrom;
use std::fmt;

use crate::basic_block::{BBFunction, BBProgram, BasicBlock};
use bril_rs::Instruction;

use fxhash::FxHashMap;

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
  ExpectedPointerType(bril_rs::Type),         // found type
  BadFuncArgType(bril_rs::Type, String),      // (expected, actual)
  BadAsmtType(bril_rs::Type, bril_rs::Type), // (expected, actual). For when the LHS type of an instruction is bad
  BadValueType(bril_rs::Type, bril_rs::Type), // (expected, actual)
  IoError(Box<std::io::Error>),
}

#[derive(Default)]
struct Environment<'a> {
  env: FxHashMap<&'a str, Value>,
}

impl <'a> Environment<'a> {
  #[inline(always)]
  pub fn get(&self, ident: &str) -> Result<&Value, InterpError> {
    self
      .env
      .get(ident)
      .ok_or_else(|| InterpError::VarNotFound(ident.to_string()))
  }
  #[inline(always)]
  pub fn set(&mut self, ident: &'a str, val: Value) {
    self.env.insert(ident, val);
  }
}

#[derive(Default)]
struct Heap {
  memory: FxHashMap<usize, Vec<Value>>,
  base_num_counter: usize,
}

impl Heap {
  #[inline(always)]
  fn is_empty(&self) -> bool {
    self.memory.is_empty()
  }

  fn alloc(&mut self, amount: i64, ptr_type: bril_rs::Type) -> Result<Value, InterpError> {
    if amount < 0 {
      return Err(InterpError::CannotAllocSize(amount));
    }
    let base = self.base_num_counter;
    self.base_num_counter += 1;
    self
      .memory
      .insert(base, vec![Value::default(); amount as usize]);
    Ok(Value::Pointer(Pointer {
      base,
      offset: 0,
      ptr_type,
    }))
  }

  fn free(&mut self, key: Pointer) -> Result<(), InterpError> {
    if self.memory.remove(&key.base).is_some() && key.offset == 0 {
      Ok(())
    } else {
      Err(InterpError::IllegalFree(key.base, key.offset))
    }
  }

  fn write(&mut self, key: &Pointer, val: Value) -> Result<(), InterpError> {
    match self.memory.get_mut(&key.base) {
      Some(vec) if vec.len() > (key.offset as usize) && key.offset >= 0 => {
        vec[key.offset as usize] = val;
        Ok(())
      }
      Some(_) | None => Err(InterpError::InvalidMemoryAccess(key.base, key.offset)),
    }
  }

  fn read(&self, key: &Pointer) -> Result<&Value, InterpError> {
    self
      .memory
      .get(&key.base)
      .and_then(|vec| vec.get(key.offset as usize))
      .ok_or(InterpError::InvalidMemoryAccess(key.base, key.offset))
      .and_then(|val| match val {
        Value::Uninitialized => Err(InterpError::UsingUninitializedMemory),
        _ => Ok(val),
      })
  }
}

fn check_asmt_type(expected: &bril_rs::Type, actual: &bril_rs::Type) -> Result<(), InterpError> {
  if expected == actual {
    Ok(())
  } else {
    Err(InterpError::BadAsmtType(expected.clone(), actual.clone()))
  }
}

#[inline(always)]
fn get_value<'a>(
  vars: &'a Environment,
  index: usize,
  args: &[String],
) -> Result<&'a Value, InterpError> {
  if index >= args.len() {
    return Err(InterpError::BadNumArgs(index, args.len()));
  }

  vars.get(&args[index])
}

#[inline(always)]
fn get_arg<'a, T>(vars: &'a Environment, index: usize, args: &[String]) -> Result<T, InterpError>
where
  T: TryFrom<&'a Value, Error = InterpError>,
{
  if index >= args.len() {
    return Err(InterpError::BadNumArgs(index + 1, args.len()));
  }

  T::try_from(vars.get(&args[index])?)
}

fn get_ptr_type(typ: &bril_rs::Type) -> Result<&bril_rs::Type, InterpError> {
  match typ {
    bril_rs::Type::Pointer(ptr_type) => Ok(&ptr_type),
    _ => Err(InterpError::ExpectedPointerType(typ.clone())),
  }
}

#[derive(Debug, Clone)]
pub enum Value {
  Int(i64),
  Bool(bool),
  Float(f64),
  Pointer(Pointer),
  Uninitialized,
}

impl Default for Value {
  fn default() -> Self {
    Value::Uninitialized
  }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Pointer {
  base: usize,
  offset: i64,
  ptr_type: bril_rs::Type,
}

impl Pointer {
  fn add(&self, offset: i64) -> Pointer {
    Pointer {
      base: self.base,
      offset: self.offset + offset,
      ptr_type: self.ptr_type.clone(),
    }
  }
  fn get_type(&self) -> &bril_rs::Type {
    let Pointer { ptr_type, .. } = self;
    ptr_type
  }
}

impl Value {
  #[inline(always)]
  pub fn get_type(&self) -> bril_rs::Type {
    match self {
      Value::Int(_) => bril_rs::Type::Int,
      Value::Bool(_) => bril_rs::Type::Bool,
      Value::Float(_) => bril_rs::Type::Float,
      Value::Pointer(Pointer { ptr_type, .. }) => {
        bril_rs::Type::Pointer(Box::new(ptr_type.clone()))
      }
      Value::Uninitialized => unreachable!(),
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
      Pointer(p) => write!(f, "{:?}", p),
      Uninitialized => unreachable!(),
    }
  }
}

impl From<&bril_rs::Literal> for Value {
  #[inline(always)]
  fn from(l: &bril_rs::Literal) -> Value {
    match l {
      bril_rs::Literal::Int(i) => Value::Int(*i),
      bril_rs::Literal::Bool(b) => Value::Bool(*b),
      bril_rs::Literal::Float(f) => Value::Float(*f),
    }
  }
}

impl From<bril_rs::Literal> for Value {
  #[inline(always)]
  fn from(l: bril_rs::Literal) -> Value {
    match l {
      bril_rs::Literal::Int(i) => Value::Int(i),
      bril_rs::Literal::Bool(b) => Value::Bool(b),
      bril_rs::Literal::Float(f) => Value::Float(f),
    }
  }
}

impl TryFrom<&Value> for i64 {
  type Error = InterpError;
  #[inline(always)]
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
  #[inline(always)]
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
  #[inline(always)]
  fn try_from(value: &Value) -> Result<Self, Self::Error> {
    if let Value::Float(f) = value {
      Ok(*f)
    } else {
      Err(InterpError::BadValueType(
        bril_rs::Type::Float,
        value.get_type(),
      ))
    }
  }
}

impl TryFrom<&Value> for Pointer {
  type Error = InterpError;
  #[inline(always)]
  fn try_from(value: &Value) -> Result<Self, Self::Error> {
    if let Value::Pointer(p) = value {
      Ok(p.clone())
    } else {
      Err(InterpError::BadValueType(
        //TODO Not sure how to get the expected type here
        bril_rs::Type::Pointer(Box::new(bril_rs::Type::Int)),
        value.get_type(),
      ))
    }
  }
}

// todo do this with less function arguments
#[allow(clippy::float_cmp)]
fn execute_value_op<'a, T: std::io::Write>(
  prog: &'a BBProgram,
  op: &bril_rs::ValueOps,
  dest: &'a str,
  op_type: &bril_rs::Type,
  args: &[String],
  labels: &[String],
  funcs: &[String],
  out: &mut T,
  value_store: &mut Environment<'a>,
  heap: &mut Heap,
  last_label: &Option<&String>,
  instruction_count: &mut u32,
) -> Result<(), InterpError> {
  use bril_rs::ValueOps::*;
  match *op {
    Add => {
      check_asmt_type(&bril_rs::Type::Int, op_type)?;
      let arg0 = get_arg::<i64>(value_store, 0, args)?;
      let arg1 = get_arg::<i64>(value_store, 1, args)?;
      value_store.set(dest, Value::Int(arg0.wrapping_add(arg1)));
    }
    Mul => {
      check_asmt_type(&bril_rs::Type::Int, op_type)?;
      let arg0 = get_arg::<i64>(value_store, 0, args)?;
      let arg1 = get_arg::<i64>(value_store, 1, args)?;
      value_store.set(dest, Value::Int(arg0.wrapping_mul(arg1)));
    }
    Sub => {
      check_asmt_type(&bril_rs::Type::Int, op_type)?;
      let arg0 = get_arg::<i64>(value_store, 0, args)?;
      let arg1 = get_arg::<i64>(value_store, 1, args)?;
      value_store.set(dest, Value::Int(arg0.wrapping_sub(arg1)));
    }
    Div => {
      check_asmt_type(&bril_rs::Type::Int, op_type)?;
      let arg0 = get_arg::<i64>(value_store, 0, args)?;
      let arg1 = get_arg::<i64>(value_store, 1, args)?;
      value_store.set(dest, Value::Int(arg0.wrapping_div(arg1)));
    }
    Eq => {
      check_asmt_type(&bril_rs::Type::Bool, op_type)?;
      let arg0 = get_arg::<i64>(value_store, 0, args)?;
      let arg1 = get_arg::<i64>(value_store, 1, args)?;
      value_store.set(dest, Value::Bool(arg0 == arg1));
    }
    Lt => {
      check_asmt_type(&bril_rs::Type::Bool, op_type)?;
      let arg0 = get_arg::<i64>(value_store, 0, args)?;
      let arg1 = get_arg::<i64>(value_store, 1, args)?;
      value_store.set(dest, Value::Bool(arg0 < arg1));
    }
    Gt => {
      check_asmt_type(&bril_rs::Type::Bool, op_type)?;
      let arg0 = get_arg::<i64>(value_store, 0, args)?;
      let arg1 = get_arg::<i64>(value_store, 1, args)?;
      value_store.set(dest, Value::Bool(arg0 > arg1));
    }
    Le => {
      check_asmt_type(&bril_rs::Type::Bool, op_type)?;
      let arg0 = get_arg::<i64>(value_store, 0, args)?;
      let arg1 = get_arg::<i64>(value_store, 1, args)?;
      value_store.set(dest, Value::Bool(arg0 <= arg1));
    }
    Ge => {
      check_asmt_type(&bril_rs::Type::Bool, op_type)?;
      let arg0 = get_arg::<i64>(value_store, 0, args)?;
      let arg1 = get_arg::<i64>(value_store, 1, args)?;
      value_store.set(dest, Value::Bool(arg0 >= arg1));
    }
    Not => {
      check_asmt_type(&bril_rs::Type::Bool, op_type)?;
      let arg0 = get_arg::<bool>(value_store, 0, args)?;
      value_store.set(dest, Value::Bool(!arg0));
    }
    And => {
      check_asmt_type(&bril_rs::Type::Bool, op_type)?;
      let arg0 = get_arg::<bool>(value_store, 0, args)?;
      let arg1 = get_arg::<bool>(value_store, 1, args)?;
      value_store.set(dest, Value::Bool(arg0 && arg1));
    }
    Or => {
      check_asmt_type(&bril_rs::Type::Bool, op_type)?;
      let arg0 = get_arg::<bool>(value_store, 0, args)?;
      let arg1 = get_arg::<bool>(value_store, 1, args)?;
      value_store.set(dest, Value::Bool(arg0 || arg1));
    }
    Id => {
      let src = get_value(value_store, 0, args)?.clone();
      check_asmt_type(op_type, &src.get_type())?;
      value_store.set(dest, src);
    }
    Fadd => {
      check_asmt_type(&bril_rs::Type::Float, op_type)?;
      let arg0 = get_arg::<f64>(value_store, 0, args)?;
      let arg1 = get_arg::<f64>(value_store, 1, args)?;
      value_store.set(dest, Value::Float(arg0 + arg1));
    }
    Fmul => {
      check_asmt_type(&bril_rs::Type::Float, op_type)?;
      let arg0 = get_arg::<f64>(value_store, 0, args)?;
      let arg1 = get_arg::<f64>(value_store, 1, args)?;
      value_store.set(dest, Value::Float(arg0 * arg1));
    }
    Fsub => {
      check_asmt_type(&bril_rs::Type::Float, op_type)?;
      let arg0 = get_arg::<f64>(value_store, 0, args)?;
      let arg1 = get_arg::<f64>(value_store, 1, args)?;
      value_store.set(dest, Value::Float(arg0 - arg1));
    }
    Fdiv => {
      check_asmt_type(&bril_rs::Type::Float, op_type)?;
      let arg0 = get_arg::<f64>(value_store, 0, args)?;
      let arg1 = get_arg::<f64>(value_store, 1, args)?;
      value_store.set(dest, Value::Float(arg0 / arg1));
    }
    Feq => {
      check_asmt_type(&bril_rs::Type::Bool, op_type)?;
      let arg0 = get_arg::<f64>(value_store, 0, args)?;
      let arg1 = get_arg::<f64>(value_store, 1, args)?;
      value_store.set(dest, Value::Bool(arg0 == arg1));
    }
    Flt => {
      check_asmt_type(&bril_rs::Type::Bool, op_type)?;
      let arg0 = get_arg::<f64>(value_store, 0, args)?;
      let arg1 = get_arg::<f64>(value_store, 1, args)?;
      value_store.set(dest, Value::Bool(arg0 < arg1));
    }
    Fgt => {
      check_asmt_type(&bril_rs::Type::Bool, op_type)?;
      let arg0 = get_arg::<f64>(value_store, 0, args)?;
      let arg1 = get_arg::<f64>(value_store, 1, args)?;
      value_store.set(dest, Value::Bool(arg0 > arg1));
    }
    Fle => {
      check_asmt_type(&bril_rs::Type::Bool, op_type)?;
      let arg0 = get_arg::<f64>(value_store, 0, args)?;
      let arg1 = get_arg::<f64>(value_store, 1, args)?;
      value_store.set(dest, Value::Bool(arg0 <= arg1));
    }
    Fge => {
      check_asmt_type(&bril_rs::Type::Bool, op_type)?;
      let arg0 = get_arg::<f64>(value_store, 0, args)?;
      let arg1 = get_arg::<f64>(value_store, 1, args)?;
      value_store.set(dest, Value::Bool(arg0 >= arg1));
    }
    Call => {
      if funcs.len() != 1 {
        return Err(InterpError::BadNumFuncs(1, funcs.len()));
      }
      let callee_func = prog
        .get(&funcs[0])
        .ok_or_else(|| InterpError::FuncNotFound(funcs[0].clone()))?;

      let next_env = make_func_args(callee_func, args, value_store)?;
      match callee_func.return_type.as_ref() {
        None => return Err(InterpError::EmptyRetForfunc(callee_func.name.clone())),
        Some(t) => check_asmt_type(op_type, t)?,
      }
      value_store.set(
        dest,
        execute(prog, callee_func, out, next_env, heap, instruction_count)?.unwrap(),
      )
    }
    Phi => {
      if args.len() != labels.len() {
        return Err(InterpError::UnequalPhiNode);
      } else if last_label.is_none() {
        return Err(InterpError::NoLastLabel);
      } else {
        let arg = labels
          .iter()
          .position(|l| l == last_label.unwrap())
          .ok_or_else(|| InterpError::PhiMissingLabel(last_label.unwrap().to_string()))
          .and_then(|i| value_store.get(args.get(i).unwrap()))?
          .clone();
        check_asmt_type(op_type, &arg.get_type())?;
        value_store.set(dest, arg);
      }
    }
    Alloc => {
      let arg0 = get_arg::<i64>(value_store, 0, args)?;
      let res = heap.alloc(arg0, get_ptr_type(op_type)?.clone())?;
      check_asmt_type(op_type, &res.get_type())?;
      value_store.set(dest, res)
    }
    Load => {
      let arg0 = get_arg::<Pointer>(value_store, 0, args)?;
      let res = heap.read(&arg0)?;
      check_asmt_type(op_type, &res.get_type())?;
      value_store.set(dest, res.clone())
    }
    PtrAdd => {
      let arg0 = get_arg::<Pointer>(value_store, 0, args)?;
      let arg1 = get_arg::<i64>(value_store, 1, args)?;
      let res = Value::Pointer(arg0.add(arg1));
      check_asmt_type(op_type, &res.get_type())?;
      value_store.set(dest, res)
    }
  }
  Ok(())
}

fn check_num_labels(expected: usize, labels: &[String]) -> Result<(), InterpError> {
  if expected != labels.len() {
    Err(InterpError::BadNumLabels(expected, labels.len()))
  } else {
    Ok(())
  }
}

// Returns a map from function parameter names to values of the call arguments
// that are bound to those parameters.
fn make_func_args<'a>(
  callee_func: &'a BBFunction,
  args: &[String],
  vars: &Environment<'a>,
) -> Result<Environment<'a>, InterpError> {
  let mut next_env = Environment::default();
  if args.is_empty() && callee_func.args.is_empty() {
    // do nothing because we have not args to add to the environment
  } else if args.len() != callee_func.args.len() {
    return Err(InterpError::BadNumArgs(callee_func.args.len(), args.len()));
  } else {
    args
      .iter()
      .zip(callee_func.args.iter())
      .try_for_each(|(arg_name, expected_arg)| {
        let arg = vars.get(arg_name)?;
        check_asmt_type(&expected_arg.arg_type, &arg.get_type())?;
        next_env.set(&expected_arg.name, arg.clone());
        Ok(())
      })?
  }

  Ok(next_env)
}

// todo do this with less function arguments
fn execute_effect_op<'a, T: std::io::Write>(
  prog: &'a BBProgram,
  func: &BBFunction,
  op: &bril_rs::EffectOps,
  args: &[String],
  labels: &[String],
  funcs: &[String],
  curr_block: &BasicBlock,
  out: &mut T,
  value_store: &Environment<'a>,
  heap: &mut Heap,
  next_block_idx: &mut Option<usize>,
  instruction_count: &mut u32,
) -> Result<Option<Value>, InterpError> {
  use bril_rs::EffectOps::*;
  match op {
    Jump => {
      check_num_labels(1, labels)?;
      *next_block_idx = Some(curr_block.exit[0]);
    }
    Branch => {
      let bool_arg0 = get_arg::<bool>(value_store, 0, args)?;
      check_num_labels(2, labels)?;
      let exit_idx = if bool_arg0 { 0 } else { 1 };
      *next_block_idx = Some(curr_block.exit[exit_idx]);
    }
    Return => match &func.return_type {
      Some(t) => {
        let arg0 = get_value(value_store, 0, args)?;
        check_asmt_type(t, &arg0.get_type())?;
        return Ok(Some(arg0.clone()));
      }
      None => {
        if args.is_empty() {
          return Ok(None);
        } else {
          return Err(InterpError::NonEmptyRetForfunc(func.name.clone()));
        }
      }
    },
    Print => {
      writeln!(
        out,
        "{}",
        args
          .iter()
          .map(|a| value_store.get(a).map(|x| format!("{}", x)))
          .collect::<Result<Vec<String>, InterpError>>()?
          .join(" ")
      )
      .map_err(|e| InterpError::IoError(Box::new(e)))?;
      out.flush().map_err(|e| InterpError::IoError(Box::new(e)))?;
    }
    Nop => {}
    Call => {
      if funcs.len() != 1 {
        return Err(InterpError::BadNumFuncs(1, funcs.len()));
      }
      let callee_func = prog
        .get(&funcs[0])
        .ok_or_else(|| InterpError::FuncNotFound(funcs[0].clone()))?;

      let next_env = make_func_args(callee_func, args, value_store)?;

      if callee_func.return_type.is_some() {
        return Err(InterpError::NonEmptyRetForfunc(callee_func.name.clone()));
      }
      if execute(prog, callee_func, out, next_env, heap, instruction_count)?.is_some() {
        unreachable!()
      }
    }
    Store => {
      let arg0 = get_arg::<Pointer>(value_store, 0, args)?;
      let arg1 = get_value(value_store, 1, args)?;
      check_asmt_type(arg0.get_type(), &arg1.get_type())?;
      heap.write(&arg0, arg1.clone())?
    }
    Free => {
      let arg0 = get_arg::<Pointer>(value_store, 0, args)?;
      heap.free(arg0)?
    }
    Speculate | Commit | Guard => unimplemented!(),
  }
  Ok(None)
}

fn execute<'a, T: std::io::Write>(
  prog: &'a BBProgram,
  func: &'a BBFunction,
  out: &mut T,
  mut value_store: Environment<'a>,
  heap: &mut Heap,
  instruction_count: &mut u32,
) -> Result<Option<Value>, InterpError> {
  // Map from variable name to value.
  let mut last_label;
  let mut current_label = None;
  let mut curr_block_idx = 0;
  let mut result = None;

  loop {
    let curr_block = &func.blocks[curr_block_idx];
    let curr_instrs = &curr_block.instrs;
    *instruction_count += curr_instrs.len() as u32;
    last_label = current_label;
    current_label = curr_block.label.as_ref();

    let mut next_block_idx = if curr_block.exit.len() == 1 {
      Some(curr_block.exit[0])
    } else {
      None
    };

    for code in curr_instrs {
      //println!("{:?}", code);
      match code {
        Instruction::Constant {
          op: bril_rs::ConstOps::Const,
          dest,
          const_type,
          value,
        } => {
          // Integer literals can be promoted to Floating point
          let value = if const_type == &bril_rs::Type::Float {
            match value {
              bril_rs::Literal::Int(i) => bril_rs::Literal::Float(*i as f64),
              _ => value.clone(),
            }
          } else {
            value.clone()
          };
          check_asmt_type(const_type, &value.get_type())?;
          value_store.set(&dest, Value::from(value));
        }
        Instruction::Value {
          op,
          dest,
          op_type,
          args,
          labels,
          funcs,
        } => {
          execute_value_op(
            prog,
            op,
            dest,
            op_type,
            args,
            labels,
            funcs,
            out,
            &mut value_store,
            heap,
            &last_label,
            instruction_count,
          )?;
        }
        Instruction::Effect {
          op,
          args,
          labels,
          funcs,
        } => {
          result = execute_effect_op(
            prog,
            func,
            op,
            args,
            labels,
            funcs,
            &curr_block,
            out,
            &value_store,
            heap,
            &mut next_block_idx,
            instruction_count,
          )?;
        }
      }
    }
    if let Some(idx) = next_block_idx {
      curr_block_idx = idx;
    } else {
      return Ok(result);
    }
  }
}

fn parse_args<'a>(
  mut env: Environment<'a>,
  args: &'a [bril_rs::Argument],
  inputs: Vec<&str>,
) -> Result<Environment<'a>, InterpError> {
  if args.is_empty() && inputs.is_empty() {
    Ok(env)
  } else if inputs.len() != args.len() {
    Err(InterpError::BadNumFuncArgs(args.len(), inputs.len()))
  } else {
    args
      .iter()
      .enumerate()
      .try_for_each(|(index, arg)| match arg.arg_type {
        bril_rs::Type::Bool => {
          match inputs.get(index).unwrap().parse::<bool>() {
            Err(_) => {
              return Err(InterpError::BadFuncArgType(
                bril_rs::Type::Bool,
                inputs.get(index).unwrap().to_string(),
              ))
            }
            Ok(b) => env.set(&arg.name, Value::Bool(b)),
          };
          Ok(())
        }
        bril_rs::Type::Int => {
          match inputs.get(index).unwrap().parse::<i64>() {
            Err(_) => {
              return Err(InterpError::BadFuncArgType(
                bril_rs::Type::Int,
                inputs.get(index).unwrap().to_string(),
              ))
            }
            Ok(i) => env.set(&arg.name, Value::Int(i)),
          };
          Ok(())
        }
        bril_rs::Type::Float => {
          match inputs.get(index).unwrap().parse::<f64>() {
            Err(_) => {
              return Err(InterpError::BadFuncArgType(
                bril_rs::Type::Float,
                inputs.get(index).unwrap().to_string(),
              ))
            }
            Ok(f) => env.set(&arg.name, Value::Float(f)),
          };
          Ok(())
        }
        bril_rs::Type::Pointer(..) => unreachable!(),
      })?;
    Ok(env)
  }
}

pub fn execute_main<T: std::io::Write>(
  prog: BBProgram,
  mut out: T,
  input_args: Vec<&str>,
  profiling: bool,
) -> Result<(), InterpError> {
  let main_func = prog.get("main").ok_or(InterpError::NoMainFunction)?;

  if main_func.return_type.is_some() {
    return Err(InterpError::NonEmptyRetForfunc(main_func.name.clone()));
  }

  let env = Environment::default();
  let mut heap = Heap::default();

  let value_store = parse_args(env, &main_func.args, input_args)?;

  let mut instruction_count = 0;

  execute(
    &prog,
    &main_func,
    &mut out,
    value_store,
    &mut heap,
    &mut instruction_count,
  )?;

  if !heap.is_empty() {
    return Err(InterpError::MemLeak);
  }

  if profiling {
    eprintln!("total_dyn_inst: {}", instruction_count);
  }

  Ok(())
}
