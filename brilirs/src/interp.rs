use std::fmt;

use crate::basic_block::{BBFunction, BBProgram, BasicBlock};
use crate::error::{InterpError, PositionalInterpError};
use bril_rs::Instruction;

use fxhash::FxHashMap;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use std::cmp::max;

// The Environment is the data structure used to represent the stack of the program.
// The values of all variables are store here. Each variable is represented as a number so
// each value can be store at the index of that number.
// Each function call gets allocated a "frame" which is just the offset that each variable
// should be index from for the duration of that call.
//  Call "main" pointer(frame size 3)
//  |
//  |        Call "foo" pointer(frame size 2)
//  |        |
// [a, b, c, a, b]
struct Environment {
  // Pointer into env for the start of the current frame
  current_pointer: usize,
  // Size of the current frame
  current_frame_size: usize,
  // A list of all stack pointers for valid frames on the stack
  stack_pointers: Vec<(usize, usize)>,
  // env is used like a stack. Assume it only grows
  env: Vec<Value>,
}

impl Environment {
  #[inline(always)]
  pub fn new(size: u32) -> Self {
    Self {
      current_pointer: 0,
      current_frame_size: size as usize,
      stack_pointers: Vec::new(),
      // Allocate a larger stack size so the interpreter needs to allocate less often
      env: vec![Value::default(); (max(size, 50)) as usize],
    }
  }
  #[inline(always)]
  pub fn get(&self, ident: &u32) -> &Value {
    // A bril program is well formed when, dynamically, every variable is defined before its use.
    // If this is violated, this will return Value::Uninitialized and the whole interpreter will come crashing down.
    self
      .env
      .get(self.current_pointer + *ident as usize)
      .unwrap()
  }

  // Used for getting arguments that should be passed to the current frame from the previous one
  pub fn get_from_last_frame(&self, ident: &u32) -> &Value {
    let past_pointer = self.stack_pointers.last().unwrap().0;
    self.env.get(past_pointer + *ident as usize).unwrap()
  }

  #[inline(always)]
  pub fn set(&mut self, ident: u32, val: Value) {
    self.env[self.current_pointer + ident as usize] = val;
  }
  // Push a new frame onto the stack
  pub fn push_frame(&mut self, size: u32) {
    self
      .stack_pointers
      .push((self.current_pointer, self.current_frame_size));
    self.current_pointer += self.current_frame_size;
    self.current_frame_size = size as usize;

    // Check that the stack is large enough
    if self.current_pointer + self.current_frame_size > self.env.len() {
      // We need to allocate more stack
      self.env.resize(
        max(
          self.env.len() * 4,
          self.current_pointer + self.current_frame_size,
        ),
        Value::default(),
      )
    }
  }

  // Remove a frame from the stack
  pub fn pop_frame(&mut self) {
    (self.current_pointer, self.current_frame_size) = self.stack_pointers.pop().unwrap();
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

// A getter function for when you just want the Value enum
#[inline(always)]
fn get_value<'a>(vars: &'a Environment, index: usize, args: &[u32]) -> &'a Value {
  vars.get(&args[index])
}

// A getter function for when you know what constructor of the Value enum you have and
// you just want the underlying value(like a f64).
#[inline(always)]
fn get_arg<'a, T>(vars: &'a Environment, index: usize, args: &[u32]) -> T
where
  T: From<&'a Value>,
{
  T::from(vars.get(&args[index]))
}

#[derive(Debug, Clone)]
enum Value {
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
struct Pointer {
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
      Value::Int(i) => write!(f, "{i}"),
      Value::Bool(b) => write!(f, "{b}"),
      Value::Float(v) => write!(f, "{v}"),
      Value::Pointer(p) => write!(f, "{p:?}"),
      Value::Uninitialized => unreachable!(),
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
      unreachable!()
    }
  }
}

impl From<&Value> for bool {
  #[inline(always)]
  fn from(value: &Value) -> Self {
    if let Value::Bool(b) = value {
      *b
    } else {
      unreachable!()
    }
  }
}

impl From<&Value> for f64 {
  #[inline(always)]
  fn from(value: &Value) -> Self {
    if let Value::Float(f) = value {
      *f
    } else {
      unreachable!()
    }
  }
}

impl<'a> From<&'a Value> for &'a Pointer {
  #[inline(always)]
  fn from(value: &'a Value) -> Self {
    if let Value::Pointer(p) = value {
      p
    } else {
      unreachable!()
    }
  }
}

// Sets up the Environment for the next function call with the supplied arguments
fn make_func_args<'a>(callee_func: &'a BBFunction, args: &[u32], vars: &mut Environment) {
  vars.push_frame(callee_func.num_of_vars);

  args
    .iter()
    .zip(callee_func.args_as_nums.iter())
    .for_each(|(arg_name, expected_arg)| {
      let arg = vars.get_from_last_frame(arg_name).clone();
      vars.set(*expected_arg, arg);
    })
}

// todo do this with less function arguments
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

      make_func_args(callee_func, args, value_store);

      let result = execute(prog, callee_func, out, value_store, heap, instruction_count)?.unwrap();

      value_store.pop_frame();

      value_store.set(dest, result)
    }
    Phi => match last_label {
      None => return Err(InterpError::NoLastLabel),
      Some(last_label) => {
        let arg = labels
          .iter()
          .position(|l| l == last_label)
          .ok_or_else(|| InterpError::PhiMissingLabel(last_label.to_string()))
          .map(|i| get_value(value_store, i, args))?
          .clone();
        value_store.set(dest, arg);
      }
    },
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
  value_store: &mut Environment,
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
    Return => {
      return Ok(
        func
          .return_type
          .as_ref()
          .map(|_| get_value(value_store, 0, args).clone()),
      )
    }
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
      // We call flush here in case `out` is a https://doc.rust-lang.org/std/io/struct.BufWriter.html
      // Otherwise we would expect this flush to be a nop.
      .and_then(|_| out.flush())
      .map_err(|e| InterpError::IoError(Box::new(e)))?;
    }
    Nop => {}
    Call => {
      let callee_func = prog
        .get(&funcs[0])
        .ok_or_else(|| InterpError::FuncNotFound(funcs[0].clone()))?;

      make_func_args(callee_func, args, value_store);

      execute(prog, callee_func, out, value_store, heap, instruction_count)?;
      value_store.pop_frame();
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
  value_store: &mut Environment,
  heap: &mut Heap,
  instruction_count: &mut u32,
) -> Result<Option<Value>, PositionalInterpError> {
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

    // This helps to implement fallthrough with basic blocks when there is no control flow instruction at the end of the block
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
          pos: _,
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
              bril_rs::Literal::Bool(_) => unreachable!(),
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
          pos,
        } => {
          execute_value_op(
            prog,
            op,
            numified_code.dest.unwrap(),
            &numified_code.args,
            labels,
            funcs,
            out,
            value_store,
            heap,
            last_label,
            instruction_count,
          )
          .map_err(|e| e.add_pos(*pos))?;
        }
        Instruction::Effect {
          op,
          args: _,
          labels: _,
          funcs,
          pos,
        } => {
          result = execute_effect_op(
            prog,
            func,
            op,
            &numified_code.args,
            funcs,
            curr_block,
            out,
            value_store,
            heap,
            &mut next_block_idx,
            instruction_count,
          )
          .map_err(|e| e.add_pos(*pos))?;
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
        bril_rs::Type::Pointer(..) => unreachable!(),
      })?;
    Ok(env)
  }
}

/// The entrance point to the interpreter. It runs over a ```prog```:[`BBProgram`] starting at the "main" function with ```input_args``` as input. Print statements output to ```out``` which implements [std::io::Write]. You also need to include whether you want the interpreter to count the number of instructions run with ```profiling```. This information is outputted to [std::io::stderr]
pub fn execute_main<T: std::io::Write, U: std::io::Write>(
  prog: &BBProgram,
  mut out: T,
  input_args: &[String],
  profiling: bool,
  mut profiling_out: U,
) -> Result<(), PositionalInterpError> {
  let main_func = prog
    .get("main")
    .ok_or_else(|| PositionalInterpError::new(InterpError::NoMainFunction))?;

  if main_func.return_type.is_some() {
    return Err(InterpError::NonEmptyRetForFunc(main_func.name.clone()))
      .map_err(|e| e.add_pos(main_func.pos));
  }

  let env = Environment::new(main_func.num_of_vars);
  let mut heap = Heap::default();

  let mut value_store = parse_args(env, &main_func.args, &main_func.args_as_nums, input_args)
    .map_err(|e| e.add_pos(main_func.pos))?;

  let mut instruction_count = 0;

  execute(
    prog,
    main_func,
    &mut out,
    &mut value_store,
    &mut heap,
    &mut instruction_count,
  )?;

  if !heap.is_empty() {
    return Err(InterpError::MemLeak).map_err(|e| e.add_pos(main_func.pos));
  }

  if profiling {
    writeln!(profiling_out, "total_dyn_inst: {instruction_count}")
      // We call flush here in case `profiling_out` is a https://doc.rust-lang.org/std/io/struct.BufWriter.html
      // Otherwise we would expect this flush to be a nop.
      .and_then(|_| profiling_out.flush())
      .map_err(|e| PositionalInterpError::new(InterpError::IoError(Box::new(e))))?;
  }

  Ok(())
}
