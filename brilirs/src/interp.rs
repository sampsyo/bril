use crate::basic_block::{BBFunction, BBProgram, BasicBlock};
use crate::error::{InterpError, PositionalInterpError};
use bril2json::escape_control_chars;
use bril_rs::Instruction;
use fxhash::FxHashMap;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use std::cmp::max;
use std::fmt;

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
#[derive(Clone)]
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
  pub fn new(size: usize) -> Self {
    Self {
      current_pointer: 0,
      current_frame_size: size,
      stack_pointers: Vec::new(),
      // Allocate a larger stack size so the interpreter needs to allocate less often
      env: vec![Value::default(); max(size, 50)],
    }
  }

  pub fn get(&self, ident: usize) -> &Value {
    // A bril program is well formed when, dynamically, every variable is defined before its use.
    // If this is violated, this will return Value::Uninitialized and the whole interpreter will come crashing down.
    self.env.get(self.current_pointer + ident).unwrap()
  }

  // Used for getting arguments that should be passed to the current frame from the previous one
  pub fn get_from_last_frame(&self, ident: usize) -> &Value {
    let past_pointer = self.stack_pointers.last().unwrap().0;
    self.env.get(past_pointer + ident).unwrap()
  }

  pub fn set(&mut self, ident: usize, val: Value) {
    self.env[self.current_pointer + ident] = val;
  }
  // Push a new frame onto the stack
  pub fn push_frame(&mut self, size: usize) {
    self
      .stack_pointers
      .push((self.current_pointer, self.current_frame_size));
    self.current_pointer += self.current_frame_size;
    self.current_frame_size = size;

    // Check that the stack is large enough
    if self.current_pointer + self.current_frame_size > self.env.len() {
      // We need to allocate more stack
      self.env.resize(
        max(
          self.env.len() * 4,
          self.current_pointer + self.current_frame_size,
        ),
        Value::default(),
      );
    }
  }

  // Remove a frame from the stack
  pub fn pop_frame(&mut self) {
    (self.current_pointer, self.current_frame_size) = self.stack_pointers.pop().unwrap();
  }
}

//A second heap for atomics, since AtomicI64 cannot be added to the Value enum without a lot of work
struct Atomics {
  atomics: HashMap<usize, AtomicI64>,
  index: AtomicUsize,
}

impl Default for Atomics {
  fn default() -> Self {
    Self {
      atomics: HashMap::new(),
      index: AtomicUsize::new(0),
    }
  }
}

impl Atomics {
  fn new_atomic(&mut self, val: i64) -> Value {
    let index = self.index.fetch_add(1, Ordering::SeqCst);
    self.atomics.insert(index, AtomicI64::new(val));
    Value::Atomic(index)
  }
  fn load_atomic(&self, index: usize) -> i64 {
    self.atomics.get(&index).unwrap().load(Ordering::SeqCst)
  }
  fn swap_atomic(&self, index: usize, val: i64) -> i64 {
    self
      .atomics
      .get(&index)
      .unwrap()
      .swap(val, Ordering::SeqCst)
  }
  fn compare_and_swap(&self, index: usize, old: i64, new: i64) -> i64 {
    match self.atomics.get(&index).unwrap().compare_exchange(
      old,
      new,
      Ordering::SeqCst,
      Ordering::SeqCst,
    ) {
      Ok(val) => val,
      Err(val) => val,
    }
  }
}

// todo: This is basically a copy of the heap implement in brili and we could probably do something smarter. This currently isn't that worth it to optimize because most benchmarks do not use the memory extension nor do they run for very long. You (the reader in the future) may be working with bril programs that you would like to speed up that extensively use the bril memory extension. In that case, it would be worth seeing how to implement Heap without a map based memory. Maybe try to re-implement malloc for a large Vec<Value>?
struct Heap {
  memory: FxHashMap<usize, Vec<Value>>,
  base_num_counter: AtomicUsize,
}

impl Default for Heap {
  fn default() -> Self {
    Self {
      memory: FxHashMap::with_capacity_and_hasher(20, fxhash::FxBuildHasher::default()),
      base_num_counter: AtomicUsize::new(0),
    }
  }
}

impl Heap {
  fn is_empty(&self) -> bool {
    self.memory.is_empty()
  }

  fn alloc(&mut self, amount: i64) -> Result<Value, InterpError> {
    let amount: usize = amount
      .try_into()
      .map_err(|_| InterpError::CannotAllocSize(amount))?;

    //make sure we atomically fetch and add the base index
    let base = self.base_num_counter.fetch_add(1, Ordering::SeqCst);
    self.memory.insert(base, vec![Value::default(); amount]);
    Ok(Value::Pointer(Pointer { base, offset: 0 }))
  }

  fn free(&mut self, key: &Pointer) -> Result<(), InterpError> {
    if self.memory.remove(&key.base).is_some() && key.offset == 0 {
      Ok(())
    } else {
      Err(InterpError::IllegalFree(key.base, key.offset))
    }
  }

  fn write(&mut self, key: &Pointer, val: Value) -> Result<(), InterpError> {
    // Will check that key.offset is >=0
    let offset: usize = key
      .offset
      .try_into()
      .map_err(|_| InterpError::InvalidMemoryAccess(key.base, key.offset))?;
    match self.memory.get_mut(&key.base) {
      Some(vec) if vec.len() > offset => {
        vec[offset] = val;
        Ok(())
      }
      Some(_) | None => Err(InterpError::InvalidMemoryAccess(key.base, key.offset)),
    }
  }

  fn read(&self, key: &Pointer) -> Result<&Value, InterpError> {
    // Will check that key.offset is >=0
    let offset: usize = key
      .offset
      .try_into()
      .map_err(|_| InterpError::InvalidMemoryAccess(key.base, key.offset))?;
    self
      .memory
      .get(&key.base)
      .and_then(|vec| vec.get(offset))
      .ok_or(InterpError::InvalidMemoryAccess(key.base, key.offset))
      .and_then(|val| match val {
        Value::Uninitialized => Err(InterpError::UsingUninitializedMemory),
        _ => Ok(val),
      })
  }
}

// A getter function for when you know what constructor of the Value enum you have and
// you just want the underlying value(like a f64).
// Or can just be used to get a owned version of the Value
fn get_arg<'a, T: From<&'a Value>>(vars: &'a Environment, index: usize, args: &[usize]) -> T {
  T::from(vars.get(args[index]))
}

#[derive(Debug, Default, Clone, Copy)]
enum Value {
  Int(i64),
  Bool(bool),
  Float(f64),
  Char(char),
  Pointer(Pointer),
  Atomic(usize),
  #[default]
  Uninitialized,
}
#[derive(Debug, Clone, PartialEq, Copy)]
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
      Self::Int(i) => write!(f, "{i}"),
      Self::Bool(b) => write!(f, "{b}"),
      Self::Float(v) if v.is_infinite() && v.is_sign_positive() => write!(f, "Infinity"),
      Self::Float(v) if v.is_infinite() && v.is_sign_negative() => write!(f, "-Infinity"),
      Self::Float(v) => write!(f, "{v:.17}"),
      Self::Char(c) => write!(f, "{c}"),
      Self::Pointer(p) => write!(f, "{p:?}"),
      Self::Atomic(a) => write!(f, "atomic_{a}"),
      Self::Uninitialized => unreachable!(),
    }
  }
}

fn optimized_val_output<T: std::io::Write>(
  out: &Mutex<T>,
  val: &Value,
) -> Result<(), std::io::Error> {
  let mut out = out.lock().unwrap();
  match val {
    Value::Int(i) => out.write_all(itoa::Buffer::new().format(*i).as_bytes()),
    Value::Bool(b) => out.write_all(if *b { b"true" } else { b"false" }),
    Value::Float(f) if f.is_infinite() && f.is_sign_positive() => out.write_all(b"Infinity"),
    Value::Float(f) if f.is_infinite() && f.is_sign_negative() => out.write_all(b"-Infinity"),
    Value::Float(f) if f.is_nan() => out.write_all(b"NaN"),
    Value::Float(f) => out.write_all(format!("{f:.17}").as_bytes()),
    Value::Char(c) => {
      let buf = &mut [0_u8; 2];
      out.write_all(c.encode_utf8(buf).as_bytes())
    }
    Value::Pointer(p) => out.write_all(format!("{p:?}").as_bytes()),
    Value::Atomic(a) => out.write_all(format!("atomic_{a}").as_bytes()),
    Value::Uninitialized => unreachable!(),
  }
}

impl From<&bril_rs::Literal> for Value {
  fn from(l: &bril_rs::Literal) -> Self {
    match l {
      bril_rs::Literal::Int(i) => Self::Int(*i),
      bril_rs::Literal::Bool(b) => Self::Bool(*b),
      bril_rs::Literal::Float(f) => Self::Float(*f),
      bril_rs::Literal::Char(c) => Self::Char(*c),
    }
  }
}

impl From<bril_rs::Literal> for Value {
  fn from(l: bril_rs::Literal) -> Self {
    match l {
      bril_rs::Literal::Int(i) => Self::Int(i),
      bril_rs::Literal::Bool(b) => Self::Bool(b),
      bril_rs::Literal::Float(f) => Self::Float(f),
      bril_rs::Literal::Char(c) => Self::Char(c),
    }
  }
}

impl From<&Value> for i64 {
  fn from(value: &Value) -> Self {
    if let Value::Int(i) = value {
      *i
    } else {
      unreachable!()
    }
  }
}

impl From<&Value> for bool {
  fn from(value: &Value) -> Self {
    if let Value::Bool(b) = value {
      *b
    } else {
      unreachable!()
    }
  }
}

impl From<&Value> for f64 {
  fn from(value: &Value) -> Self {
    if let Value::Float(f) = value {
      *f
    } else {
      unreachable!()
    }
  }
}

impl From<&Value> for char {
  fn from(value: &Value) -> Self {
    if let Value::Char(c) = value {
      *c
    } else {
      unreachable!()
    }
  }
}

impl<'a> From<&'a Value> for &'a Pointer {
  fn from(value: &'a Value) -> Self {
    if let Value::Pointer(p) = value {
      p
    } else {
      unreachable!()
    }
  }
}

impl From<&Value> for usize {
  fn from(value: &Value) -> Self {
    if let Value::Atomic(idx) = value {
      *idx
    } else {
      unreachable!()
    }
  }
}

impl From<&Self> for Value {
  fn from(value: &Self) -> Self {
    *value
  }
}

// Sets up the Environment for the next function call with the supplied arguments
fn make_func_args(callee_func: &BBFunction, args: &[usize], vars: &mut Environment) {
  vars.push_frame(callee_func.num_of_vars);

  args
    .iter()
    .zip(callee_func.args_as_nums.iter())
    .for_each(|(arg_name, expected_arg)| {
      let arg = vars.get_from_last_frame(*arg_name);
      vars.set(*expected_arg, *arg);
    });
}

fn execute_thread<T: std::io::Write + Sync + Send + 'static>(
  mut state: State,
  func: usize,
  out: Arc<Mutex<T>>,
  stop_flag: Arc<AtomicBool>,
) -> Option<Value> {
  return execute(&mut state, func, out, Some(stop_flag)).unwrap();
}

fn execute_value_op<T: std::io::Write + Send + Sync + 'static>(
  state: &mut State,
  op: bril_rs::ValueOps,
  dest: usize,
  args: &[usize],
  labels: &[String],
  funcs: &[usize],
  last_label: Option<&String>,
  handlers: &mut HashMap<usize, (JoinHandle<Option<Value>>, Arc<AtomicBool>)>,
  out: Arc<Mutex<T>>,
  stop_flag: Option<Arc<AtomicBool>>,
) -> Result<(), InterpError> {
  use bril_rs::ValueOps::{
    Add, Alloc, And, Call, Ceq, Cge, Cgt, Char2int, Cle, Clt, CompareAndSwap, Div, Eq, Fadd, Fdiv,
    Feq, Fge, Fgt, Fle, Flt, Fmul, Fsub, Ge, Gt, Id, Int2char, Le, Load, LoadAtomic, Lt, Mul,
    NewAtomic, Not, Or, Phi, PtrAdd, Resolve, Sub, SwapAtomic,
  };
  match op {
    Add => {
      let arg0 = get_arg::<i64>(&state.env, 0, args);
      let arg1 = get_arg::<i64>(&state.env, 1, args);
      state.env.set(dest, Value::Int(arg0.wrapping_add(arg1)));
    }
    Mul => {
      let arg0 = get_arg::<i64>(&state.env, 0, args);
      let arg1 = get_arg::<i64>(&state.env, 1, args);
      state.env.set(dest, Value::Int(arg0.wrapping_mul(arg1)));
    }
    Sub => {
      let arg0 = get_arg::<i64>(&state.env, 0, args);
      let arg1 = get_arg::<i64>(&state.env, 1, args);
      state.env.set(dest, Value::Int(arg0.wrapping_sub(arg1)));
    }
    Div => {
      let arg0 = get_arg::<i64>(&state.env, 0, args);
      let arg1 = get_arg::<i64>(&state.env, 1, args);
      if arg1 == 0 {
        return Err(InterpError::DivisionByZero);
      }
      state.env.set(dest, Value::Int(arg0.wrapping_div(arg1)));
    }
    Eq => {
      let arg0 = get_arg::<i64>(&state.env, 0, args);
      let arg1 = get_arg::<i64>(&state.env, 1, args);
      state.env.set(dest, Value::Bool(arg0 == arg1));
    }
    Lt => {
      let arg0 = get_arg::<i64>(&state.env, 0, args);
      let arg1 = get_arg::<i64>(&state.env, 1, args);
      state.env.set(dest, Value::Bool(arg0 < arg1));
    }
    Gt => {
      let arg0 = get_arg::<i64>(&state.env, 0, args);
      let arg1 = get_arg::<i64>(&state.env, 1, args);
      state.env.set(dest, Value::Bool(arg0 > arg1));
    }
    Le => {
      let arg0 = get_arg::<i64>(&state.env, 0, args);
      let arg1 = get_arg::<i64>(&state.env, 1, args);
      state.env.set(dest, Value::Bool(arg0 <= arg1));
    }
    Ge => {
      let arg0 = get_arg::<i64>(&state.env, 0, args);
      let arg1 = get_arg::<i64>(&state.env, 1, args);
      state.env.set(dest, Value::Bool(arg0 >= arg1));
    }
    Not => {
      let arg0 = get_arg::<bool>(&state.env, 0, args);
      state.env.set(dest, Value::Bool(!arg0));
    }
    And => {
      let arg0 = get_arg::<bool>(&state.env, 0, args);
      let arg1 = get_arg::<bool>(&state.env, 1, args);
      state.env.set(dest, Value::Bool(arg0 && arg1));
    }
    Or => {
      let arg0 = get_arg::<bool>(&state.env, 0, args);
      let arg1 = get_arg::<bool>(&state.env, 1, args);
      state.env.set(dest, Value::Bool(arg0 || arg1));
    }
    Id => {
      let src = get_arg::<Value>(&state.env, 0, args);
      state.env.set(dest, src);
    }
    Fadd => {
      let arg0 = get_arg::<f64>(&state.env, 0, args);
      let arg1 = get_arg::<f64>(&state.env, 1, args);
      state.env.set(dest, Value::Float(arg0 + arg1));
    }
    Fmul => {
      let arg0 = get_arg::<f64>(&state.env, 0, args);
      let arg1 = get_arg::<f64>(&state.env, 1, args);
      state.env.set(dest, Value::Float(arg0 * arg1));
    }
    Fsub => {
      let arg0 = get_arg::<f64>(&state.env, 0, args);
      let arg1 = get_arg::<f64>(&state.env, 1, args);
      state.env.set(dest, Value::Float(arg0 - arg1));
    }
    Fdiv => {
      let arg0 = get_arg::<f64>(&state.env, 0, args);
      let arg1 = get_arg::<f64>(&state.env, 1, args);
      state.env.set(dest, Value::Float(arg0 / arg1));
    }
    Feq => {
      let arg0 = get_arg::<f64>(&state.env, 0, args);
      let arg1 = get_arg::<f64>(&state.env, 1, args);
      state.env.set(dest, Value::Bool(arg0 == arg1));
    }
    Flt => {
      let arg0 = get_arg::<f64>(&state.env, 0, args);
      let arg1 = get_arg::<f64>(&state.env, 1, args);
      state.env.set(dest, Value::Bool(arg0 < arg1));
    }
    Fgt => {
      let arg0 = get_arg::<f64>(&state.env, 0, args);
      let arg1 = get_arg::<f64>(&state.env, 1, args);
      state.env.set(dest, Value::Bool(arg0 > arg1));
    }
    Fle => {
      let arg0 = get_arg::<f64>(&state.env, 0, args);
      let arg1 = get_arg::<f64>(&state.env, 1, args);
      state.env.set(dest, Value::Bool(arg0 <= arg1));
    }
    Fge => {
      let arg0 = get_arg::<f64>(&state.env, 0, args);
      let arg1 = get_arg::<f64>(&state.env, 1, args);
      state.env.set(dest, Value::Bool(arg0 >= arg1));
    }
    Ceq => {
      let arg0 = get_arg::<char>(&state.env, 0, args);
      let arg1 = get_arg::<char>(&state.env, 1, args);
      state.env.set(dest, Value::Bool(arg0 == arg1));
    }
    Clt => {
      let arg0 = get_arg::<char>(&state.env, 0, args);
      let arg1 = get_arg::<char>(&state.env, 1, args);
      state.env.set(dest, Value::Bool(arg0 < arg1));
    }
    Cgt => {
      let arg0 = get_arg::<char>(&state.env, 0, args);
      let arg1 = get_arg::<char>(&state.env, 1, args);
      state.env.set(dest, Value::Bool(arg0 > arg1));
    }
    Cle => {
      let arg0 = get_arg::<char>(&state.env, 0, args);
      let arg1 = get_arg::<char>(&state.env, 1, args);
      state.env.set(dest, Value::Bool(arg0 <= arg1));
    }
    Cge => {
      let arg0 = get_arg::<char>(&state.env, 0, args);
      let arg1 = get_arg::<char>(&state.env, 1, args);
      state.env.set(dest, Value::Bool(arg0 >= arg1));
    }
    Char2int => {
      let arg0 = get_arg::<char>(&state.env, 0, args);
      state.env.set(dest, Value::Int(u32::from(arg0).into()));
    }
    Int2char => {
      let arg0 = get_arg::<i64>(&state.env, 0, args);

      let arg0_char = u32::try_from(arg0)
        .ok()
        .and_then(char::from_u32)
        .ok_or(InterpError::ToCharError(arg0))?;

      state.env.set(dest, Value::Char(arg0_char));
    }
    Call => {
      let callee_func = state.prog.get(funcs[0]).unwrap();

      if callee_func.is_promise() {
        let mut cloned = state.clone();
        make_func_args(callee_func, args, &mut cloned.env);

        let callee_func = funcs[0];
        let out = out.clone();

        let thread_stop_flag = Arc::new(AtomicBool::new(false));
        let cloned_stop_flag = thread_stop_flag.clone();
        handlers.insert(
          dest,
          (
            thread::spawn(move || execute_thread(cloned, callee_func, out, cloned_stop_flag)),
            thread_stop_flag,
          ),
        );
      } else {
        make_func_args(callee_func, args, &mut state.env);

        let result = execute(state, funcs[0].clone(), out, stop_flag)?;

        let val = match result {
          Some(val) => val,
          None => Value::Uninitialized,
        };

        state.env.pop_frame();
        state.env.set(dest, val);
      }
    }
    Phi => match last_label {
      None => return Err(InterpError::NoLastLabel),
      Some(last_label) => {
        let arg = labels
          .iter()
          .position(|l| l == last_label)
          .ok_or_else(|| InterpError::PhiMissingLabel(last_label.to_string()))
          .map(|i| get_arg::<Value>(&state.env, i, args))?;
        state.env.set(dest, arg);
      }
    },
    Alloc => {
      let arg0 = get_arg::<i64>(&state.env, 0, args);
      //allows for multiple threads to (unsafely) access the heap at once
      //synchronization should be accomplished using the implemented atomics
      //heap base pointer is atomically incremented so allocation is still thread safe
      let heap = Arc::into_raw(state.heap.clone()) as *mut Heap;
      let res = unsafe { (*heap).alloc(arg0)? };
      let _ = unsafe { Arc::from_raw(heap) };

      state.env.set(dest, res);
    }
    Load => {
      let arg0 = get_arg::<&Pointer>(&state.env, 0, args);
      let res = state.heap.read(arg0)?;
      state.env.set(dest, *res);
    }
    PtrAdd => {
      let arg0 = get_arg::<&Pointer>(&state.env, 0, args);
      let arg1 = get_arg::<i64>(&state.env, 1, args);
      let res = Value::Pointer(arg0.add(arg1));
      state.env.set(dest, res);
    }
    Resolve => {
      let res = match handlers.remove(&args[0]) {
        Some(handle) => handle.0.join().unwrap(),
        None => panic!("Unexpected execution error"),
      };

      state.env.set(dest, res.unwrap());
    }
    CompareAndSwap => {
      //atomic
      let arg0 = get_arg::<usize>(&state.env, 0, args);
      //expected
      let arg1 = get_arg::<i64>(&state.env, 1, args);
      //update
      let arg2 = get_arg::<i64>(&state.env, 2, args);
      let res = state.atomics.compare_and_swap(arg0, arg1, arg2);
      state.env.set(dest, Value::Int(res));
    }
    LoadAtomic => {
      let arg0 = get_arg::<usize>(&state.env, 0, args) as usize;
      let res = state.atomics.load_atomic(arg0);
      state.env.set(dest, Value::Int(res));
    }
    SwapAtomic => {
      let arg0 = get_arg::<usize>(&state.env, 0, args) as usize;
      let arg1 = get_arg::<i64>(&state.env, 1, args);
      let res = state.atomics.swap_atomic(arg0, arg1);
      state.env.set(dest, Value::Int(res));
    }
    NewAtomic => {
      let arg0 = get_arg::<i64>(&state.env, 0, args);
      let atomics = Arc::into_raw(state.atomics.clone()) as *mut Atomics;
      let res = unsafe { (*atomics).new_atomic(arg0) };
      let _ = unsafe { Arc::from_raw(atomics) };

      state.env.set(dest, res);
    }
  }
  Ok(())
}

fn execute_effect_op<T: std::io::Write + Sync + Send + 'static>(
  state: &mut State,
  op: bril_rs::EffectOps,
  args: &[usize],
  funcs: &[usize],
  curr_block: &BasicBlock,
  // There are two output variables where values are stored to effect the loop execution.
  next_block_idx: &mut Option<usize>,
  result: &mut Option<Value>,
  out: Arc<Mutex<T>>,
  stop_flag: Option<Arc<AtomicBool>>,
) -> Result<(), InterpError> {
  use bril_rs::EffectOps::{
    Branch, Call, Commit, Free, Guard, Jump, Nop, Print, Return, Speculate, Store,
  };
  match op {
    Jump => {
      *next_block_idx = Some(curr_block.exit[0]);
    }
    Branch => {
      let bool_arg0 = get_arg::<bool>(&state.env, 0, args);
      let exit_idx = usize::from(!bool_arg0);
      *next_block_idx = Some(curr_block.exit[exit_idx]);
    }
    Return => {
      if !args.is_empty() {
        *result = Some(get_arg::<Value>(&state.env, 0, args));
      }
    }
    Print => {
      // In the typical case, users only print out one value at a time
      // So we can usually avoid extra allocations by providing that string directly
      {
        if args.len() == 1 {
          optimized_val_output(&out, state.env.get(*args.first().unwrap()))?;
          // Add new line
          let mut out = out.lock().unwrap();
          out.write_all(&[b'\n'])?;
        } else {
          let mut out = out.lock().unwrap();
          writeln!(
            out,
            "{}",
            args
              .iter()
              .map(|a| state.env.get(*a).to_string())
              .collect::<Vec<String>>()
              .join(" ")
          )?;
        }
      }
    }
    Nop => {}
    Call => {
      let callee_func = state.prog.get(funcs[0]).unwrap();

      make_func_args(callee_func, args, &mut state.env);

      execute(state, funcs[0].clone(), out, stop_flag)?;
      state.env.pop_frame();
    }
    Store => {
      let arg0 = get_arg::<&Pointer>(&state.env, 0, args);
      let arg1 = get_arg::<Value>(&state.env, 1, args);
      let heap = Arc::into_raw(state.heap.clone()) as *mut Heap;
      unsafe { (*heap).write(arg0, arg1)? };
      let _ = unsafe { Arc::from_raw(heap) };
    }
    Free => {
      let arg0 = get_arg::<&Pointer>(&state.env, 0, args);
      let heap = Arc::into_raw(state.heap.clone()) as *mut Heap;
      unsafe { (*heap).free(arg0)? };
      let _ = unsafe { Arc::from_raw(heap) };
    }
    Speculate | Commit | Guard => unimplemented!(),
  }
  Ok(())
}

fn cleanup_handlers(handlers: HashMap<usize, (JoinHandle<Option<Value>>, Arc<AtomicBool>)>) {
  for (_, (handle, stop_flag)) in handlers {
    stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = handle.join();
  }
}

fn execute<T: std::io::Write + Sync + Send + 'static>(
  state: &mut State,
  func: usize,
  out: Arc<Mutex<T>>,
  stop_flag: Option<Arc<AtomicBool>>,
) -> Result<Option<Value>, PositionalInterpError> {
  let func = state.prog.get(func).unwrap().clone();
  let mut last_label;
  let mut current_label = None;
  let mut curr_block_idx = 0;
  // A possible return value
  let mut result = None;

  let mut handlers = HashMap::new();

  loop {
    //checks if the thread should be terminated
    if let Some(stop_flag) = &stop_flag {
      if stop_flag.load(std::sync::atomic::Ordering::Relaxed) {
        cleanup_handlers(handlers);
        panic!("Terminating unresolved thread. This may lead to undefined behavior.");
      }
    };
    let curr_block = &func.blocks[curr_block_idx];
    let curr_instrs = &curr_block.instrs;
    let curr_numified_instrs = &curr_block.numified_instrs;
    // WARNING!!! We can add the # of instructions at once because you can only jump to a new block at the end. This may need to be changed if speculation is implemented
    //state.instruction_count += curr_instrs.len();
    state
      .instruction_count
      .fetch_add(curr_instrs.len(), std::sync::atomic::Ordering::Relaxed);
    last_label = current_label;
    current_label = curr_block.label.as_ref();

    // A place to store the next block that will be jumped to if specified by an instruction
    let mut next_block_idx = None;

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
              // So yes, as clippy points out, you technically lose precision here on the `*i as f64` cast. On the other hand, you already give up precision when you start using floats and I haven't been able to find a case where you are giving up precision in the cast that you don't already lose by using floating points.
              // So it's probably fine unless proven otherwise.
              #[allow(clippy::cast_precision_loss)]
              bril_rs::Literal::Int(i) => state
                .env
                .set(numified_code.dest.unwrap(), Value::Float(*i as f64)),
              bril_rs::Literal::Float(f) => {
                state.env.set(numified_code.dest.unwrap(), Value::Float(*f));
              }
              bril_rs::Literal::Char(_) | bril_rs::Literal::Bool(_) => unreachable!(),
            }
          } else {
            state
              .env
              .set(numified_code.dest.unwrap(), Value::from(value));
          };
        }
        Instruction::Value {
          op,
          dest: _,
          op_type: _,
          args: _,
          labels,
          funcs: _,
          pos,
        } => {
          execute_value_op(
            state,
            *op,
            numified_code.dest.unwrap(),
            &numified_code.args,
            labels,
            &numified_code.funcs,
            last_label,
            &mut handlers,
            out.clone(),
            stop_flag.clone(),
          )
          .map_err(|e| e.add_pos(pos.clone()))?;
        }
        Instruction::Effect {
          op,
          args: _,
          labels: _,
          funcs: _,
          pos,
        } => {
          execute_effect_op(
            state,
            *op,
            &numified_code.args,
            &numified_code.funcs,
            curr_block,
            &mut next_block_idx,
            &mut result,
            out.clone(),
            stop_flag.clone(),
          )
          .map_err(|e| e.add_pos(pos.clone()))?;
        }
      }
    }

    // Are we jumping to a new block or are we done?
    if let Some(idx) = next_block_idx {
      curr_block_idx = idx;
    } else if curr_block.exit.len() == 1 {
      curr_block_idx = curr_block.exit[0];
    } else {
      cleanup_handlers(handlers);
      return Ok(result);
    }
  }
}

fn parse_args(
  mut env: Environment,
  args: &[bril_rs::Argument],
  args_as_nums: &[usize],
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
        bril_rs::Type::Promise(..) => unreachable!("line 839"),
        bril_rs::Type::AtomicInt => unreachable!("line 840"),
        bril_rs::Type::Char => escape_control_chars(inputs.get(index).unwrap().as_ref())
          .map_or_else(
            || Err(InterpError::NotOneChar),
            |c| {
              env.set(*arg_as_num, Value::Char(c));
              Ok(())
            },
          ),
      })?;
    Ok(env)
  }
}

#[derive(Clone)]
// State captures the parts of the interpreter that are used across function boundaries
struct State {
  prog: Arc<BBProgram>,
  env: Environment,
  heap: Arc<Heap>,
  atomics: Arc<Atomics>,
  instruction_count: Arc<AtomicUsize>,
}

impl State {
  fn new(prog: BBProgram, env: Environment, heap: Heap) -> Self {
    Self {
      prog: Arc::new(prog),
      env,
      heap: Arc::new(heap),
      atomics: Arc::new(Atomics::default()),
      instruction_count: Arc::new(AtomicUsize::new(0)),
    }
  }
}

/// The entrance point to the interpreter. It runs over a ```prog```:[`BBProgram`] starting at the "main" function with ```input_args``` as input. Print statements output to ```out``` which implements [`std::io::Write`]. You also need to include whether you want the interpreter to count the number of instructions run with ```profiling```. This information is outputted to [`std::io::stderr`]
/// # Panics
/// This should not panic with normal use except if there is a bug or if you are using an unimplemented feature
/// # Errors
/// Will error on malformed `BBProgram`, like if the original Bril program was not well-formed
pub fn execute_main<T: std::io::Write + Sync + Send + 'static, U: std::io::Write>(
  prog: BBProgram,
  out: Mutex<T>,
  input_args: &[String],
  profiling: bool,
  mut profiling_out: U,
) -> Result<(), PositionalInterpError> {
  let main_func_idx = match prog.index_of_main {
    Some(i) => i,
    None => return Err(InterpError::NoMainFunction.into()),
  };

  let main_func = prog.get(main_func_idx).unwrap();
  let mut env = Environment::new(main_func.num_of_vars);
  let heap = Heap::default();
  let main_func_pos = main_func.pos.clone();

  env = parse_args(env, &main_func.args, &main_func.args_as_nums, input_args)
    .map_err(|e| e.add_pos(main_func.pos.clone()))?;

  let mut state = State::new(prog, env, heap);

  let out = Arc::new(out);
  execute(&mut state, main_func_idx, out.clone(), None)?;

  if !state.heap.is_empty() {
    return Err(InterpError::MemLeak).map_err(|e| e.add_pos(main_func_pos));
  }

  {
    let mut writer = out.lock().unwrap();
    writer.flush().map_err(InterpError::IoError)?;
  }

  if profiling {
    writeln!(
      profiling_out,
      "total_dyn_inst: {}",
      state
        .instruction_count
        .load(std::sync::atomic::Ordering::Relaxed)
    )
    // We call flush here in case `profiling_out` is a https://doc.rust-lang.org/std/io/struct.BufWriter.html
    // Otherwise we would expect this flush to be a nop.
    .and_then(|()| profiling_out.flush())
    .map_err(InterpError::IoError)?;
  }

  Ok(())
}
