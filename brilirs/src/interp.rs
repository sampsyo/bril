use std::fmt;
use std::hint::unreachable_unchecked;

use crate::basic_block::{BBFunction, BBProgram, BasicBlock};
use crate::error::InterpError;
use bril_rs::Instruction;

use fxhash::FxHashMap;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

struct Environment {
  env: Vec<Value>,
}

impl Environment {
  #[inline(always)]
  pub fn new(size: u32) -> Self {
    Self {
      env: vec![Value::default(); size as usize],
    }
  }
  #[inline(always)]
  pub fn get(&self, ident: &u32) -> &Value {
    // A bril program is well formed when, dynamically, every variable is defined before its use.
    // If this is violated, this will return Value::Uninitialized and the whole interpreter will come crashing down.
    self.env.get(*ident as usize).unwrap()
  }
  #[inline(always)]
  pub fn set(&mut self, ident: u32, val: Value) {
    self.env[ident as usize] = val;
  }
}

// todo: This is basically a copy of the heap implement in brili and we could probably do something smarter. This currently isn't that worth it to optimize because most benchmarks do not use the memory extension nor do they run for very long. You (the reader in the future) may be working with bril programs that you would like to speed up that extensively use the bril memory extension. In that case, it would be worth seeing how to implement Heap without a map based memory. Maybe try to re-implement malloc for a large Vec<Value>?
struct Heap {
  memory: FxHashMap<usize, Vec<Value>>,
  base_num_counter: usize,
}

impl Default for Heap {
  fn default() -> Self {
    Self {
      memory: FxHashMap::with_capacity_and_hasher(20, fxhash::FxBuildHasher::default()),
      base_num_counter: 0,
    }
  }
}

impl Heap {
  #[inline(always)]
  fn is_empty(&self) -> bool {
    self.memory.is_empty()
  }

  #[inline(always)]
  fn alloc(&mut self, amount: i64) -> Result<Value, InterpError> {
    if amount < 0 {
      return Err(InterpError::CannotAllocSize(amount));
    }
    let base = self.base_num_counter;
    self.base_num_counter += 1;
    self
      .memory
      .insert(base, vec![Value::default(); amount as usize]);
    Ok(Value::Pointer(Pointer { base, offset: 0 }))
  }

  #[inline(always)]
  fn free(&mut self, key: &Pointer) -> Result<(), InterpError> {
    if self.memory.remove(&key.base).is_some() && key.offset == 0 {
      Ok(())
    } else {
      Err(InterpError::IllegalFree(key.base, key.offset))
    }
  }

  #[inline(always)]
  fn write(&mut self, key: &Pointer, val: Value) -> Result<(), InterpError> {
    match self.memory.get_mut(&key.base) {
      Some(vec) if vec.len() > (key.offset as usize) && key.offset >= 0 => {
        vec[key.offset as usize] = val;
        Ok(())
      }
      Some(_) | None => Err(InterpError::InvalidMemoryAccess(key.base, key.offset)),
    }
  }

  #[inline(always)]
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

#[inline(always)]
fn get_value<'a>(vars: &'a Environment, index: usize, args: &[u32]) -> &'a Value {
  vars.get(&args[index])
}

#[inline(always)]
fn get_arg<'a, T>(vars: &'a Environment, index: usize, args: &[u32]) -> T
where
  T: From<&'a Value>,
{
  T::from(vars.get(&args[index]))
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
    Self::Uninitialized
  }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Pointer {
  base: usize,
  offset: i64,
}

impl Pointer {
  const fn add(&self, offset: i64) -> Self {
    Self {
      base: self.base,
      offset: self.offset + offset,
    }
  }
}

impl fmt::Display for Value {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Value::Int(i) => write!(f, "{}", i),
      Value::Bool(b) => write!(f, "{}", b),
      Value::Float(v) => write!(f, "{}", v),
      Value::Pointer(p) => write!(f, "{:?}", p),
      // This is safe because Uninitialized is only used in relation to memory and immediately errors if this value is returned. Otherwise this value can not appear in the code
      Value::Uninitialized => unsafe { unreachable_unchecked() },
    }
  }
}

impl From<&bril_rs::Literal> for Value {
  #[inline(always)]
  fn from(l: &bril_rs::Literal) -> Self {
    match l {
      bril_rs::Literal::Int(i) => Self::Int(*i),
      bril_rs::Literal::Bool(b) => Self::Bool(*b),
      bril_rs::Literal::Float(f) => Self::Float(*f),
    }
  }
}

impl From<bril_rs::Literal> for Value {
  #[inline(always)]
  fn from(l: bril_rs::Literal) -> Self {
    match l {
      bril_rs::Literal::Int(i) => Self::Int(i),
      bril_rs::Literal::Bool(b) => Self::Bool(b),
      bril_rs::Literal::Float(f) => Self::Float(f),
    }
  }
}

impl From<&Value> for i64 {
  #[inline(always)]
  fn from(value: &Value) -> Self {
    if let Value::Int(i) = value {
      *i
    } else {
      // This is safe because we type check the program beforehand
      unsafe { unreachable_unchecked() }
    }
  }
}

impl From<&Value> for bool {
  #[inline(always)]
  fn from(value: &Value) -> Self {
    if let Value::Bool(b) = value {
      *b
    } else {
      // This is safe because we type check the program beforehand
      unsafe { unreachable_unchecked() }
    }
  }
}

impl From<&Value> for f64 {
  #[inline(always)]
  fn from(value: &Value) -> Self {
    if let Value::Float(f) = value {
      *f
    } else {
      // This is safe because we type check the program beforehand
      unsafe { unreachable_unchecked() }
    }
  }
}

impl<'a> From<&'a Value> for &'a Pointer {
  #[inline(always)]
  fn from(value: &'a Value) -> Self {
    if let Value::Pointer(p) = value {
      p
    } else {
      // This is safe because we type check the program beforehand
      unsafe { unreachable_unchecked() }
    }
  }
}

// todo do this with less function arguments
#[allow(clippy::float_cmp)]
#[inline(always)]
fn execute_value_op<'a, T: std::io::Write>(
  prog: &'a BBProgram,
  op: &bril_rs::ValueOps,
  dest: u32,
  args: &[u32],
  labels: &[String],
  funcs: &[String],
  out: &mut T,
  value_store: &mut Environment,
  heap: &mut Heap,
  last_label: Option<&String>,
  instruction_count: &mut u32,
) -> Result<(), InterpError> {
  use bril_rs::ValueOps::*;
  match *op {
    Add => {
      let arg0 = get_arg::<i64>(value_store, 0, args);
      let arg1 = get_arg::<i64>(value_store, 1, args);
      value_store.set(dest, Value::Int(arg0.wrapping_add(arg1)));
    }
    Mul => {
      let arg0 = get_arg::<i64>(value_store, 0, args);
      let arg1 = get_arg::<i64>(value_store, 1, args);
      value_store.set(dest, Value::Int(arg0.wrapping_mul(arg1)));
    }
    Sub => {
      let arg0 = get_arg::<i64>(value_store, 0, args);
      let arg1 = get_arg::<i64>(value_store, 1, args);
      value_store.set(dest, Value::Int(arg0.wrapping_sub(arg1)));
    }
    Div => {
      let arg0 = get_arg::<i64>(value_store, 0, args);
      let arg1 = get_arg::<i64>(value_store, 1, args);
      value_store.set(dest, Value::Int(arg0.wrapping_div(arg1)));
    }
    Eq => {
      let arg0 = get_arg::<i64>(value_store, 0, args);
      let arg1 = get_arg::<i64>(value_store, 1, args);
      value_store.set(dest, Value::Bool(arg0 == arg1));
    }
    Lt => {
      let arg0 = get_arg::<i64>(value_store, 0, args);
      let arg1 = get_arg::<i64>(value_store, 1, args);
      value_store.set(dest, Value::Bool(arg0 < arg1));
    }
    Gt => {
      let arg0 = get_arg::<i64>(value_store, 0, args);
      let arg1 = get_arg::<i64>(value_store, 1, args);
      value_store.set(dest, Value::Bool(arg0 > arg1));
    }
    Le => {
      let arg0 = get_arg::<i64>(value_store, 0, args);
      let arg1 = get_arg::<i64>(value_store, 1, args);
      value_store.set(dest, Value::Bool(arg0 <= arg1));
    }
    Ge => {
      let arg0 = get_arg::<i64>(value_store, 0, args);
      let arg1 = get_arg::<i64>(value_store, 1, args);
      value_store.set(dest, Value::Bool(arg0 >= arg1));
    }
    Not => {
      let arg0 = get_arg::<bool>(value_store, 0, args);
      value_store.set(dest, Value::Bool(!arg0));
    }
    And => {
      let arg0 = get_arg::<bool>(value_store, 0, args);
      let arg1 = get_arg::<bool>(value_store, 1, args);
      value_store.set(dest, Value::Bool(arg0 && arg1));
    }
    Or => {
      let arg0 = get_arg::<bool>(value_store, 0, args);
      let arg1 = get_arg::<bool>(value_store, 1, args);
      value_store.set(dest, Value::Bool(arg0 || arg1));
    }
    Id => {
      let src = get_value(value_store, 0, args).clone();
      value_store.set(dest, src);
    }
    Fadd => {
      let arg0 = get_arg::<f64>(value_store, 0, args);
      let arg1 = get_arg::<f64>(value_store, 1, args);
      value_store.set(dest, Value::Float(arg0 + arg1));
    }
    Fmul => {
      let arg0 = get_arg::<f64>(value_store, 0, args);
      let arg1 = get_arg::<f64>(value_store, 1, args);
      value_store.set(dest, Value::Float(arg0 * arg1));
    }
    Fsub => {
      let arg0 = get_arg::<f64>(value_store, 0, args);
      let arg1 = get_arg::<f64>(value_store, 1, args);
      value_store.set(dest, Value::Float(arg0 - arg1));
    }
    Fdiv => {
      let arg0 = get_arg::<f64>(value_store, 0, args);
      let arg1 = get_arg::<f64>(value_store, 1, args);
      value_store.set(dest, Value::Float(arg0 / arg1));
    }
    Feq => {
      let arg0 = get_arg::<f64>(value_store, 0, args);
      let arg1 = get_arg::<f64>(value_store, 1, args);
      value_store.set(dest, Value::Bool(arg0 == arg1));
    }
    Flt => {
      let arg0 = get_arg::<f64>(value_store, 0, args);
      let arg1 = get_arg::<f64>(value_store, 1, args);
      value_store.set(dest, Value::Bool(arg0 < arg1));
    }
    Fgt => {
      let arg0 = get_arg::<f64>(value_store, 0, args);
      let arg1 = get_arg::<f64>(value_store, 1, args);
      value_store.set(dest, Value::Bool(arg0 > arg1));
    }
    Fle => {
      let arg0 = get_arg::<f64>(value_store, 0, args);
      let arg1 = get_arg::<f64>(value_store, 1, args);
      value_store.set(dest, Value::Bool(arg0 <= arg1));
    }
    Fge => {
      let arg0 = get_arg::<f64>(value_store, 0, args);
      let arg1 = get_arg::<f64>(value_store, 1, args);
      value_store.set(dest, Value::Bool(arg0 >= arg1));
    }
    Call => {
      let callee_func = prog
        .get(&funcs[0])
        .ok_or_else(|| InterpError::FuncNotFound(funcs[0].clone()))?;

      let next_env = make_func_args(callee_func, args, value_store);

      value_store.set(
        dest,
        execute(prog, callee_func, out, next_env, heap, instruction_count)?.unwrap(),
      )
    }
    Phi => {
      if last_label.is_none() {
        return Err(InterpError::NoLastLabel);
      } else {
        let arg = labels
          .iter()
          .position(|l| l == last_label.unwrap())
          .ok_or_else(|| InterpError::PhiMissingLabel(last_label.unwrap().to_string()))
          .map(|i| value_store.get(args.get(i).unwrap()))?
          .clone();
        value_store.set(dest, arg);
      }
    }
    Alloc => {
      let arg0 = get_arg::<i64>(value_store, 0, args);
      let res = heap.alloc(arg0)?;
      value_store.set(dest, res)
    }
    Load => {
      let arg0 = get_arg::<&Pointer>(value_store, 0, args);
      let res = heap.read(arg0)?;
      value_store.set(dest, res.clone())
    }
    PtrAdd => {
      let arg0 = get_arg::<&Pointer>(value_store, 0, args);
      let arg1 = get_arg::<i64>(value_store, 1, args);
      let res = Value::Pointer(arg0.add(arg1));
      value_store.set(dest, res)
    }
  }
  Ok(())
}

// Returns a map from function parameter names to values of the call arguments
// that are bound to those parameters.
fn make_func_args<'a>(
  callee_func: &'a BBFunction,
  args: &[u32],
  vars: &Environment,
) -> Environment {
  // todo: Having to allocate a new environment on each function call probably makes small function calls very heavy weight. This could be interesting to profile and see if old environments can be reused instead of being deallocated and reallocated. Maybe there is another way to sometimes avoid this allocation.
  let mut next_env = Environment::new(callee_func.num_of_vars);

  args
    .iter()
    .zip(callee_func.args_as_nums.iter())
    .for_each(|(arg_name, expected_arg)| {
      let arg = vars.get(arg_name);
      next_env.set(*expected_arg, arg.clone());
    });

  next_env
}

// todo do this with less function arguments
#[inline(always)]
fn execute_effect_op<'a, T: std::io::Write>(
  prog: &'a BBProgram,
  func: &BBFunction,
  op: &bril_rs::EffectOps,
  args: &[u32],
  funcs: &[String],
  curr_block: &BasicBlock,
  out: &mut T,
  value_store: &Environment,
  heap: &mut Heap,
  next_block_idx: &mut Option<usize>,
  instruction_count: &mut u32,
) -> Result<Option<Value>, InterpError> {
  use bril_rs::EffectOps::*;
  match op {
    Jump => {
      *next_block_idx = Some(curr_block.exit[0]);
    }
    Branch => {
      let bool_arg0 = get_arg::<bool>(value_store, 0, args);
      let exit_idx = if bool_arg0 { 0 } else { 1 };
      *next_block_idx = Some(curr_block.exit[exit_idx]);
    }
    Return => match &func.return_type {
      Some(_) => {
        let arg0 = get_value(value_store, 0, args);
        return Ok(Some(arg0.clone()));
      }
      None => return Ok(None),
    },
    Print => {
      writeln!(
        out,
        "{}",
        args
          .iter()
          .map(|a| value_store.get(a).to_string())
          .collect::<Vec<String>>()
          .join(" ")
      )
      .map_err(|e| InterpError::IoError(Box::new(e)))?;
      out.flush().map_err(|e| InterpError::IoError(Box::new(e)))?;
    }
    Nop => {}
    Call => {
      let callee_func = prog
        .get(&funcs[0])
        .ok_or_else(|| InterpError::FuncNotFound(funcs[0].clone()))?;

      let next_env = make_func_args(callee_func, args, value_store);

      execute(prog, callee_func, out, next_env, heap, instruction_count)?;
    }
    Store => {
      let arg0 = get_arg::<&Pointer>(value_store, 0, args);
      let arg1 = get_value(value_store, 1, args);
      heap.write(arg0, arg1.clone())?
    }
    Free => {
      let arg0 = get_arg::<&Pointer>(value_store, 0, args);
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
  mut value_store: Environment,
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
    let curr_numified_instrs = &curr_block.numified_instrs;
    // WARNING!!! We can add the # of instructions at once because you can only jump to a new block at the end. This may need to be changed if speculation is implemented
    *instruction_count += curr_instrs.len() as u32;
    last_label = current_label;
    current_label = curr_block.label.as_ref();

    let mut next_block_idx = if curr_block.exit.len() == 1 {
      Some(curr_block.exit[0])
    } else {
      None
    };

    for (code, numified_code) in curr_instrs.iter().zip(curr_numified_instrs.iter()) {
      match code {
        Instruction::Constant {
          op: bril_rs::ConstOps::Const,
          dest: _,
          const_type,
          value,
        } => {
          // Integer literals can be promoted to Floating point
          if const_type == &bril_rs::Type::Float {
            match value {
              bril_rs::Literal::Int(i) => {
                value_store.set(numified_code.dest.unwrap(), Value::Float(*i as f64))
              }
              bril_rs::Literal::Float(f) => {
                value_store.set(numified_code.dest.unwrap(), Value::Float(*f))
              }
              // this is safe because we type check this beforehand
              bril_rs::Literal::Bool(_) => unsafe { unreachable_unchecked() },
            }
          } else {
            value_store.set(numified_code.dest.unwrap(), Value::from(value));
          };
        }
        Instruction::Value {
          op,
          dest: _,
          op_type: _,
          args: _,
          labels,
          funcs,
        } => {
          execute_value_op(
            prog,
            op,
            numified_code.dest.unwrap(),
            &numified_code.args,
            labels,
            funcs,
            out,
            &mut value_store,
            heap,
            last_label,
            instruction_count,
          )?;
        }
        Instruction::Effect {
          op,
          args: _,
          labels: _,
          funcs,
        } => {
          result = execute_effect_op(
            prog,
            func,
            op,
            &numified_code.args,
            funcs,
            curr_block,
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

fn parse_args(
  mut env: Environment,
  args: &[bril_rs::Argument],
  args_as_nums: &[u32],
  inputs: &[String],
) -> Result<Environment, InterpError> {
  if args.is_empty() && inputs.is_empty() {
    Ok(env)
  } else if inputs.len() != args.len() {
    Err(InterpError::BadNumFuncArgs(args.len(), inputs.len()))
  } else {
    args
      .iter()
      .zip(args_as_nums.iter())
      .enumerate()
      .try_for_each(|(index, (arg, arg_as_num))| match arg.arg_type {
        bril_rs::Type::Bool => {
          match inputs.get(index).unwrap().parse::<bool>() {
            Err(_) => {
              return Err(InterpError::BadFuncArgType(
                bril_rs::Type::Bool,
                (*inputs.get(index).unwrap()).to_string(),
              ))
            }
            Ok(b) => env.set(*arg_as_num, Value::Bool(b)),
          };
          Ok(())
        }
        bril_rs::Type::Int => {
          match inputs.get(index).unwrap().parse::<i64>() {
            Err(_) => {
              return Err(InterpError::BadFuncArgType(
                bril_rs::Type::Int,
                (*inputs.get(index).unwrap()).to_string(),
              ))
            }
            Ok(i) => env.set(*arg_as_num, Value::Int(i)),
          };
          Ok(())
        }
        bril_rs::Type::Float => {
          match inputs.get(index).unwrap().parse::<f64>() {
            Err(_) => {
              return Err(InterpError::BadFuncArgType(
                bril_rs::Type::Float,
                (*inputs.get(index).unwrap()).to_string(),
              ))
            }
            Ok(f) => env.set(*arg_as_num, Value::Float(f)),
          };
          Ok(())
        }
        // this is safe because there is no possible way to pass a pointer as an argument
        bril_rs::Type::Pointer(..) => unsafe { unreachable_unchecked() },
      })?;
    Ok(env)
  }
}

pub fn execute_main<T: std::io::Write>(
  prog: &BBProgram,
  mut out: T,
  input_args: &[String],
  profiling: bool,
) -> Result<(), InterpError> {
  let main_func = prog.get("main").ok_or(InterpError::NoMainFunction)?;

  if main_func.return_type.is_some() {
    return Err(InterpError::NonEmptyRetForFunc(main_func.name.clone()));
  }

  let env = Environment::new(main_func.num_of_vars);
  let mut heap = Heap::default();

  let value_store = parse_args(env, &main_func.args, &main_func.args_as_nums, input_args)?;

  let mut instruction_count = 0;

  execute(
    prog,
    main_func,
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
