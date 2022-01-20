use bril_rs::{Function, Instruction, Program};
use error::InterpError;
use fxhash::FxHashMap;

use crate::error;

// A program represented as basic blocks.
#[derive(Debug)]
pub struct BBProgram {
  pub func_index: FxHashMap<String, BBFunction>,
}

impl BBProgram {
  pub fn new(prog: Program) -> Result<Self, InterpError> {
    let num_funcs = prog.functions.len();
    let bb = Self {
      func_index: prog
        .functions
        .into_iter()
        .map(|func| (func.name.clone(), BBFunction::new(func)))
        .collect(),
    };
    if bb.func_index.len() != num_funcs {
      Err(InterpError::DuplicateFunction)
    } else {
      Ok(bb)
    }
  }

  pub fn get(&self, func_name: &str) -> Option<&BBFunction> {
    self.func_index.get(func_name)
  }
}

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

#[derive(Debug)]
pub struct NumifiedInstruction {
  pub dest: Option<u32>,
  pub args: Vec<u32>,
}

fn get_num_from_map(
  var: &str,
  num_of_vars: &mut u32,
  num_var_map: &mut FxHashMap<String, u32>,
) -> u32 {
  match num_var_map.get(var) {
    Some(i) => *i,
    None => {
      let x = *num_of_vars;
      num_var_map.insert(var.to_string(), x);
      *num_of_vars += 1;
      x
    }
  }
}

impl NumifiedInstruction {
  pub fn create(
    instr: &Instruction,
    num_of_vars: &mut u32,
    num_var_map: &mut FxHashMap<String, u32>,
  ) -> Self {
    match instr {
      Instruction::Constant { dest, .. } => Self {
        dest: Some(get_num_from_map(dest, num_of_vars, num_var_map)),
        args: Vec::new(),
      },
      Instruction::Value { dest, args, .. } => Self {
        dest: Some(get_num_from_map(dest, num_of_vars, num_var_map)),
        args: args
          .iter()
          .map(|v| get_num_from_map(v, num_of_vars, num_var_map))
          .collect(),
      },
      Instruction::Effect { args, .. } => Self {
        dest: None,
        args: args
          .iter()
          .map(|v| get_num_from_map(v, num_of_vars, num_var_map))
          .collect(),
      },
    }
  }
}

#[derive(Debug)]
pub struct BBFunction {
  pub name: String,
  pub args: Vec<bril_rs::Argument>,
  pub return_type: Option<bril_rs::Type>,
  pub blocks: Vec<BasicBlock>,
  // the following is an optimization by replacing the string representation of variables with a number
  // Variable names are ordered from 0 to num_of_vars.
  // These replacements are found for function args and for code in the BasicBlocks
  pub num_of_vars: u32,
  pub args_as_nums: Vec<u32>,
}

impl BBFunction {
  pub fn new(f: Function) -> Self {
    let (mut func, label_map) = Self::find_basic_blocks(f);
    func.build_cfg(label_map);
    func
  }

  fn find_basic_blocks(func: bril_rs::Function) -> (Self, FxHashMap<String, usize>) {
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
    for instr in func.instrs.into_iter() {
      match instr {
        bril_rs::Code::Label { label } => {
          if !curr_block.instrs.is_empty() || curr_block.label.is_some() {
            if let Some(old_label) = curr_block.label.as_ref() {
              label_map.insert(old_label.to_string(), blocks.len());
            }
            blocks.push(curr_block);
            curr_block = BasicBlock::new();
          }
          curr_block.label = Some(label);
        }
        bril_rs::Code::Instruction(bril_rs::Instruction::Effect {
          op,
          args,
          funcs,
          labels,
        }) if op == bril_rs::EffectOps::Jump
          || op == bril_rs::EffectOps::Branch
          || op == bril_rs::EffectOps::Return =>
        {
          let i = bril_rs::Instruction::Effect {
            op,
            args,
            funcs,
            labels,
          };
          curr_block.numified_instrs.push(NumifiedInstruction::create(
            &i,
            &mut num_of_vars,
            &mut num_var_map,
          ));
          curr_block.instrs.push(i);
          if let Some(l) = curr_block.label.as_ref() {
            label_map.insert(l.to_string(), blocks.len());
          }
          blocks.push(curr_block);
          curr_block = BasicBlock::new();
        }
        bril_rs::Code::Instruction(code) => {
          curr_block.numified_instrs.push(NumifiedInstruction::create(
            &code,
            &mut num_of_vars,
            &mut num_var_map,
          ));
          curr_block.instrs.push(code);
        }
      }
    }

    if !curr_block.instrs.is_empty() || curr_block.label.is_some() {
      if let Some(l) = curr_block.label.as_ref() {
        label_map.insert(l.to_string(), blocks.len());
      }
      blocks.push(curr_block);
    }

    (
      Self {
        name: func.name,
        args: func.args,
        return_type: func.return_type,
        blocks,
        args_as_nums,
        num_of_vars,
      },
      label_map,
    )
  }

  fn build_cfg(&mut self, label_map: FxHashMap<String, usize>) {
    let last_idx = self.blocks.len() - 1;
    for (i, block) in self.blocks.iter_mut().enumerate() {
      // If we're before the last block
      if i < last_idx {
        // Get the last instruction
        let last_instr = block.instrs.last().cloned();
        if let Some(bril_rs::Instruction::Effect {
          op: bril_rs::EffectOps::Jump,
          labels,
          ..
        })
        | Some(bril_rs::Instruction::Effect {
          op: bril_rs::EffectOps::Branch,
          labels,
          ..
        }) = last_instr
        {
          for l in labels {
            block.exit.push(
              *label_map
                .get(&l)
                .unwrap_or_else(|| panic!("No label {} found.", &l)),
            );
          }
        } else if let Some(bril_rs::Instruction::Effect {
          op: bril_rs::EffectOps::Return,
          ..
        }) = last_instr
        {
          // We are done, there is no exit from this block
        } else {
          block.exit.push(i + 1);
        }
      }
    }
  }
}
