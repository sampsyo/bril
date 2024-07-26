use bril_rs::{Function, Instruction, Position, Program};
use fxhash::FxHashMap;

use crate::error::{InterpError, PositionalInterpError};

/// A program represented as basic blocks. This is the IR of brilirs
#[derive(Debug)]
pub struct BBProgram {
  #[doc(hidden)]
  pub index_of_main: Option<usize>,
  #[doc(hidden)]
  pub func_index: Vec<BBFunction>,
}

impl TryFrom<Program> for BBProgram {
  type Error = InterpError;

  fn try_from(prog: Program) -> Result<Self, Self::Error> {
    Self::new(prog)
  }
}

impl BBProgram {
  /// Converts a [`Program`] into a [`BBProgram`]
  /// # Errors
  /// Will return an error if the program is invalid in some way.
  /// Reasons include the `Program` have multiple functions with the same name, a function name is not found, or a label is expected by an instruction but missing.
  pub fn new(prog: Program) -> Result<Self, InterpError> {
    let num_funcs = prog.functions.len();

    let func_map: FxHashMap<String, usize> = prog
      .functions
      .iter()
      .enumerate()
      .map(|(idx, func)| (func.name.clone(), idx))
      .collect();

    let func_index = prog
      .functions
      .into_iter()
      .map(|func| BBFunction::new(func, &func_map))
      .collect::<Result<Vec<BBFunction>, InterpError>>()?;

    let bb = Self {
      index_of_main: func_map.get("main").copied(),
      func_index,
    };
    if func_map.len() == num_funcs {
      Ok(bb)
    } else {
      Err(InterpError::DuplicateFunction)
    }
  }

  #[doc(hidden)]
  #[must_use]
  pub fn get(&self, func_name: usize) -> Option<&BBFunction> {
    self.func_index.get(func_name)
  }
}

#[doc(hidden)]
#[derive(Debug)]
pub struct BasicBlock {
  pub label: Option<String>,
  // These two vecs work in parallel
  // One is the normal instruction
  // The other contains the numified version of the destination and arguments
  pub instrs: Vec<bril_rs::Instruction>,
  pub numified_instrs: Vec<NumifiedInstruction>,
  pub exit: Vec<usize>,
}

impl BasicBlock {
  const fn new() -> Self {
    Self {
      label: None,
      instrs: Vec::new(),
      numified_instrs: Vec::new(),
      exit: Vec::new(),
    }
  }
}

#[doc(hidden)]
#[derive(Debug)]
pub struct NumifiedInstruction {
  pub dest: Option<usize>,
  pub args: Vec<usize>,
  pub funcs: Vec<usize>,
}

fn get_num_from_map(
  variable_name: &str,
  // The total number of variables so far. Only grows
  num_of_vars: &mut usize,
  // A map from variables to numbers
  num_var_map: &mut FxHashMap<String, usize>,
) -> usize {
  // https://github.com/rust-lang/rust-clippy/issues/8346
  #[allow(clippy::option_if_let_else)]
  if let Some(i) = num_var_map.get(variable_name) {
    *i
  } else {
    let x = *num_of_vars;
    num_var_map.insert(variable_name.to_string(), x);
    *num_of_vars += 1;
    x
  }
}

impl NumifiedInstruction {
  fn new(
    instr: &Instruction,
    // The total number of variables so far. Only grows
    num_of_vars: &mut usize,
    // A map from variables to numbers
    num_var_map: &mut FxHashMap<String, usize>,
    // A map from function names to numbers
    func_map: &FxHashMap<String, usize>,
  ) -> Result<Self, PositionalInterpError> {
    Ok(match instr {
      Instruction::Constant { dest, .. } => Self {
        dest: Some(get_num_from_map(dest, num_of_vars, num_var_map)),
        args: Vec::new(),
        funcs: Vec::new(),
      },
      Instruction::Value {
        dest,
        args,
        funcs,
        pos,
        ..
      } => Self {
        dest: Some(get_num_from_map(dest, num_of_vars, num_var_map)),
        args: args
          .iter()
          .map(|v| get_num_from_map(v, num_of_vars, num_var_map))
          .collect(),
        funcs: funcs
          .iter()
          .map(|f| {
            func_map
              .get(f)
              .copied()
              .ok_or_else(|| InterpError::FuncNotFound(f.to_string()).add_pos(pos.clone()))
          })
          .collect::<Result<Vec<usize>, PositionalInterpError>>()?,
      },
      Instruction::Effect {
        args, funcs, pos, ..
      } => Self {
        dest: None,
        args: args
          .iter()
          .map(|v| get_num_from_map(v, num_of_vars, num_var_map))
          .collect(),
        funcs: funcs
          .iter()
          .map(|f| {
            func_map
              .get(f)
              .copied()
              .ok_or_else(|| InterpError::FuncNotFound(f.to_string()).add_pos(pos.clone()))
          })
          .collect::<Result<Vec<usize>, PositionalInterpError>>()?,
      },
    })
  }
}

#[doc(hidden)]
#[derive(Debug)]
pub struct BBFunction {
  pub name: String,
  pub args: Vec<bril_rs::Argument>,
  pub return_type: Option<bril_rs::Type>,
  pub blocks: Vec<BasicBlock>,
  // the following is an optimization by replacing the string representation of variables with a number
  // Variable names are ordered from 0 to num_of_vars.
  // These replacements are found for function args and for code in the BasicBlocks
  pub num_of_vars: usize,
  pub args_as_nums: Vec<usize>,
  pub pos: Option<Position>,
}

impl BBFunction {
  fn new(f: Function, func_map: &FxHashMap<String, usize>) -> Result<Self, InterpError> {
    let (mut func, label_map) = Self::find_basic_blocks(f, func_map)?;
    func.build_cfg(&label_map)?;
    Ok(func)
  }

  fn find_basic_blocks(
    func: bril_rs::Function,
    func_map: &FxHashMap<String, usize>,
  ) -> Result<(Self, FxHashMap<String, usize>), PositionalInterpError> {
    let mut blocks = Vec::new();
    let mut label_map = FxHashMap::default();

    let mut num_of_vars = 0;
    let mut num_var_map = FxHashMap::default();

    let args_as_nums = func
      .args
      .iter()
      .map(|a| get_num_from_map(&a.name, &mut num_of_vars, &mut num_var_map))
      .collect();

    let mut curr_block = BasicBlock::new();
    for instr in func.instrs {
      match instr {
        bril_rs::Code::Label { label, pos } => {
          if !curr_block.instrs.is_empty() || curr_block.label.is_some() {
            blocks.push(curr_block);
            curr_block = BasicBlock::new();
          }
          if label_map.insert(label.to_string(), blocks.len()).is_some() {
            return Err(InterpError::DuplicateLabel(label).add_pos(pos));
          }
          curr_block.label = Some(label);
        }
        bril_rs::Code::Instruction(i @ bril_rs::Instruction::Effect { op, .. })
          if op == bril_rs::EffectOps::Jump
            || op == bril_rs::EffectOps::Branch
            || op == bril_rs::EffectOps::Return =>
        {
          curr_block.numified_instrs.push(NumifiedInstruction::new(
            &i,
            &mut num_of_vars,
            &mut num_var_map,
            func_map,
          )?);
          curr_block.instrs.push(i);
          blocks.push(curr_block);
          curr_block = BasicBlock::new();
        }
        bril_rs::Code::Instruction(code) => {
          curr_block.numified_instrs.push(NumifiedInstruction::new(
            &code,
            &mut num_of_vars,
            &mut num_var_map,
            func_map,
          )?);
          curr_block.instrs.push(code);
        }
      }
    }

    if !curr_block.instrs.is_empty() || curr_block.label.is_some() {
      blocks.push(curr_block);
    }

    Ok((
      Self {
        name: func.name,
        args: func.args,
        return_type: func.return_type,
        blocks,
        args_as_nums,
        num_of_vars,
        pos: func.pos,
      },
      label_map,
    ))
  }

  fn build_cfg(&mut self, label_map: &FxHashMap<String, usize>) -> Result<(), InterpError> {
    if self.blocks.is_empty() {
      return Ok(());
    }
    let last_idx = self.blocks.len() - 1;
    for (i, block) in self.blocks.iter_mut().enumerate() {
      // Get the last instruction
      let last_instr = block.instrs.last().cloned();
      if let Some(bril_rs::Instruction::Effect {
        op: bril_rs::EffectOps::Jump | bril_rs::EffectOps::Branch,
        labels,
        ..
      }) = last_instr
      {
        for l in labels {
          block
            .exit
            .push(*label_map.get(&l).ok_or(InterpError::MissingLabel(l))?);
        }
      } else if let Some(bril_rs::Instruction::Effect {
        op: bril_rs::EffectOps::Return,
        ..
      }) = last_instr
      {
        // We are done, there is no exit from this block
      } else {
        // If we're before the last block
        if i < last_idx {
          block.exit.push(i + 1);
        }
      }
    }
    Ok(())
  }
}
