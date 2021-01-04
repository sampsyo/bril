use bril_rs::{Function, Program};
use std::collections::HashMap;

// A program represented as basic blocks.
pub struct BBProgram {
  pub func_index: HashMap<String, BBFunction>,
}

impl BBProgram {
  pub fn new(prog: Program) -> BBProgram {
    BBProgram {
      func_index: prog
        .functions
        .into_iter()
        .map(|func| (func.name.clone(), BBFunction::new(func)))
        .collect(),
    }
  }

  pub fn get(&self, func_name: &str) -> Option<&BBFunction> {
    self.func_index.get(func_name)
  }
}

#[derive(Debug)]
pub struct BasicBlock {
  pub label: Option<String>,
  pub instrs: Vec<bril_rs::Instruction>,
  pub exit: Vec<usize>,
}

impl BasicBlock {
  fn new() -> BasicBlock {
    BasicBlock {
      label: None,
      instrs: Vec::new(),
      exit: Vec::new(),
    }
  }
}

pub struct BBFunction {
  pub name: String,
  pub args: Vec<bril_rs::Argument>,
  pub return_type: Option<bril_rs::Type>,
  pub blocks: Vec<BasicBlock>,
  pub label_map: HashMap<String, usize>,
}

impl BBFunction {
  pub fn new(f: Function) -> BBFunction {
    let mut func = BBFunction::find_basic_blocks(f);
    func.build_cfg();
    func
  }

  fn find_basic_blocks(func: bril_rs::Function) -> BBFunction {
    let mut blocks = Vec::new();
    let mut label_map = HashMap::new();

    let mut curr_block = BasicBlock::new();
    for instr in func.instrs.into_iter() {
      match instr {
        bril_rs::Code::Label { ref label } => {
          if !curr_block.instrs.is_empty() {
            if let Some(old_label) = curr_block.label.as_ref() {
              label_map.insert(old_label.to_string(), blocks.len());
            }
            blocks.push(curr_block);
            curr_block = BasicBlock::new();
          }
          curr_block.label = Some(label.clone());
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
          curr_block.instrs.push(bril_rs::Instruction::Effect {
            op,
            args,
            funcs,
            labels,
          });
          if let Some(l) = curr_block.label.as_ref() {
            label_map.insert(l.to_string(), blocks.len());
          }
          blocks.push(curr_block);
          curr_block = BasicBlock::new();
        }
        bril_rs::Code::Instruction(code) => {
          curr_block.instrs.push(code);
        }
      }
    }

    if !curr_block.instrs.is_empty() {
      if let Some(l) = curr_block.label.as_ref() {
        label_map.insert(l.to_string(), blocks.len());
      }
      blocks.push(curr_block);
    }

    BBFunction {
      name: func.name,
      args: func.args,
      return_type: func.return_type,
      blocks,
      label_map,
    }
  }

  fn build_cfg(&mut self) {
    let last_idx = self.blocks.len() - 1;
    for (i, block) in self.blocks.iter_mut().enumerate() {
      // If we're before the last block
      if i < last_idx {
        // Get the last instruction
        if let Some(bril_rs::Instruction::Effect { op, labels, .. }) = block.instrs.last().cloned()
        {
          if let bril_rs::EffectOps::Jump | bril_rs::EffectOps::Branch = op {
            for l in labels {
              block.exit.push(
                *self
                  .label_map
                  .get(&l)
                  .unwrap_or_else(|| panic!("No label {} found.", &l)),
              );
            }
          }
        } else {
          block.exit.push(i + 1);
        }
      }
    }
  }
}
