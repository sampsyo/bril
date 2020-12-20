use crate::basic_block::BasicBlock;
use bril_rs;

use std::collections::HashMap;

type CFG = Vec<BasicBlock>;

pub fn build_cfg(mut blocks: Vec<BasicBlock>, label_to_block_idx: &HashMap<String, usize>) -> CFG {
  let last_idx = blocks.len() - 1;
  for (i, block) in blocks.iter_mut().enumerate() {
    // If we're before the last block
    if i < last_idx {
      // Get the last instruction
      let last_instr: &bril_rs::Code = block.instrs.last().unwrap();
      if let bril_rs::Code::Instruction(bril_rs::Instruction::Effect { op, labels, .. }) =
        last_instr
      {
        match op {
          bril_rs::EffectOps::Jump | bril_rs::EffectOps::Branch => {
            for l in labels {
              block.exit.push(label_to_block_idx[l]);
            }
          }
          bril_rs::EffectOps::Return => {}
          // TODO(yati): Do all effect ops end a BB?
          _ => {}
        }
      } else {
        block.exit.push(i + 1);
      }
    }
  }

  blocks
}
