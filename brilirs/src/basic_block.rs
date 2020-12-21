use std::collections::HashMap;

// A program composed of basic blocks.
// (BB index of main program, list of BBs, mapping of label -> BB index)
pub type BBProgram = (Option<usize>, Vec<BasicBlock>, HashMap<String, usize>);

#[derive(Debug)]
pub struct BasicBlock {
  pub instrs: Vec<bril_rs::Code>,
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

pub fn find_basic_blocks(prog: bril_rs::Program) -> BBProgram {
  let mut main_fn = None;
  let mut blocks = Vec::new();
  let mut labels = HashMap::new();

  let mut bb_helper = |func: bril_rs::Function| -> usize {
    let mut curr_block = BasicBlock::new();
    let root_block = blocks.len();
    let mut curr_label = None;
    for instr in func.instrs.into_iter() {
      match instr {
        bril_rs::Code::Label { ref label } => {
          if !curr_block.instrs.is_empty() {
            blocks.push(curr_block);
            if let Some(old_label) = curr_label {
              labels.insert(old_label, blocks.len() - 1);
            }
            curr_block = BasicBlock::new();
          }
          curr_label = Some(label.clone());
        }
        bril_rs::Code::Instruction(bril_rs::Instruction::Effect { op, .. })
          if op == bril_rs::EffectOps::Jump
            || op == bril_rs::EffectOps::Branch
            || op == bril_rs::EffectOps::Return =>
        {
          curr_block.instrs.push(instr);
          blocks.push(curr_block);
          if let Some(l) = curr_label {
            labels.insert(l, blocks.len() - 1);
            curr_label = None;
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
      if let Some(l) = curr_label {
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
