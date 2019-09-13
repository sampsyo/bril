use crate::ir_types::{Function, Instruction, Program};
use std::collections::HashMap;

pub struct BasicBlock {
  pub instrs: Vec<Instruction>,
  pub exit: Vec<usize>,
}

impl BasicBlock {
  fn new() -> BasicBlock {
    BasicBlock {
      instrs: Vec::new(),
      exit: Vec::new(),
    }
  }
}

pub fn find_basic_blocks(
  prog: Program,
) -> (Option<usize>, Vec<BasicBlock>, HashMap<String, usize>) {
  let mut main_fn = None;
  let mut blocks = Vec::new();
  let mut labels = HashMap::new();

  let mut bb_helper = |func: Function| -> usize {
    let mut curr_block = BasicBlock::new();
    let root_block = blocks.len();
    let mut label = None;
    for instr in func.instrs.into_iter() {
      match instr {
        Instruction::Label(ref l) => {
          if !curr_block.instrs.is_empty() {
            blocks.push(curr_block);
            if let Some(old_label) = label {
              labels.insert(old_label, blocks.len() - 1);
            }

            curr_block = BasicBlock::new();
          }

          label = Some(l.label.clone());
          curr_block.instrs.push(instr);
        }

        Instruction::Jmp { .. } | Instruction::Br { .. } | Instruction::Ret { .. } => {
          curr_block.instrs.push(instr);
          blocks.push(curr_block);
          if let Some(l) = label {
            labels.insert(l, blocks.len() - 1);
            label = None;
          }

          curr_block = BasicBlock::new();
        }
        _ => {
          curr_block.instrs.push(instr);
        }
      }
    }

    if !curr_block.instrs.is_empty() {
      blocks.push(curr_block);
      if let Some(l) = label {
        labels.insert(l, blocks.len() - 1);
      }
    }

    root_block
  };

  for func in prog.functions.into_iter() {
    let func_name = func.name.clone();
    let func_block = bb_helper(func);
    if func_name == "main" {
      main_fn = Some(func_block);
    }
  }

  (main_fn, blocks, labels)
}
