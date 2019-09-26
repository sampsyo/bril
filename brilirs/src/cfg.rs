use crate::basic_block::BasicBlock;
use crate::ir_types::{BrArgs, Operation};

use std::collections::HashMap;

type CFG = Vec<BasicBlock>;

pub fn build_cfg(mut blocks: Vec<BasicBlock>, labels: &HashMap<String, usize>) -> CFG {
  let last_idx = blocks.len() - 1;
  for (i, block) in blocks.iter_mut().enumerate() {
    // If we're before the last block
    if i < last_idx {
      // Get the last instruction
      let last_instr: &Operation<usize> = block.instrs.last().unwrap();
      match last_instr {
        // Either the last instruction is a terminal instruction...
        Operation::Jmp { params } => {
          for arg in params.args.iter() {
            block.exit.push(labels[arg]);
          }
        }
        Operation::Br {
          params: BrArgs::IdArgs { dests, .. },
        } => {
          for arg in dests.iter() {
            block.exit.push(labels[arg]);
          }
        }

        // Or it's a return, which we hack around until function calls are implemented
        Operation::Ret { .. } => {}

        // Or the first instruction of the *next* block is a label and we fall through
        _ => {
          block.exit.push(i + 1);
        }
      }
    }
  }

  blocks
}
