use std::collections::HashMap;

pub struct Function {
  pub args: Vec<bril_rs::Argument>,
  pub return_type: Option<bril_rs::Type>,
  pub blocks: Vec<BasicBlock>,

  // Map from label to the index of the block that is the target of the label.
  pub label_index: HashMap<String, usize>,
}

impl Function {
  pub fn new(f: bril_rs::Function) -> Function {
    let mut func = Function {
      args: f.args.clone(),
      return_type: f.return_type.clone(),
      blocks: vec![],
      label_index: HashMap::new(),
    };
    func.add_blocks(f.instrs);
    func.build_cfg();
    func
  }

  fn build_cfg(&mut self) {
    let last_idx = self.blocks.len() - 1;
    for (i, block) in self.blocks.iter_mut().enumerate() {
      // If we're before the last block
      if i < last_idx {
        // Get the last instruction
        let last_instr: &bril_rs::Code = block.instrs.last().unwrap();
        if let bril_rs::Code::Instruction(bril_rs::Instruction::Effect { op, labels, .. }) =
          last_instr
        {
          if let bril_rs::EffectOps::Jump | bril_rs::EffectOps::Branch = op {
            for l in labels {
              block.exit.push(
                *self
                  .label_index
                  .get(l)
                  .expect(&format!("No label {} found.", &l)),
              );
            }
          }
        } else {
          block.exit.push(i + 1);
        }
      }
    }
  }

  fn add_blocks(&mut self, instrs: Vec<bril_rs::Code>) {
    let mut curr_block = BasicBlock::new();
    let mut curr_label = None;
    for instr in instrs {
      match instr {
        bril_rs::Code::Label { ref label } => {
          if !curr_block.instrs.is_empty() {
            self.blocks.push(curr_block);
            if let Some(old_label) = curr_label {
              self.label_index.insert(old_label, self.blocks.len() - 1);
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
          self.blocks.push(curr_block);
          if let Some(l) = curr_label {
            self.label_index.insert(l, self.blocks.len() - 1);
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
      // If we are here, the function ends without an explicit ret. To make
      // processing easier, push a Return op onto the last block.
      curr_block.instrs.push(RET.clone());
      self.blocks.push(curr_block);
      if let Some(l) = curr_label {
        self.label_index.insert(l, self.blocks.len() - 1);
      }
    }
  }
}

// A program represented as basic blocks.
pub struct BBProgram {
  pub func_index: HashMap<String, Function>,
}

impl BBProgram {
  pub fn new(prog: bril_rs::Program) -> BBProgram {
    let mut bbprog = BBProgram {
      func_index: HashMap::new(),
    };
    for func in prog.functions {
      bbprog
        .func_index
        .insert(func.name.clone(), Function::new(func));
    }
    bbprog
  }
}

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

const RET: bril_rs::Code = bril_rs::Code::Instruction(bril_rs::Instruction::Effect {
  op: bril_rs::EffectOps::Return,
  args: vec![],
  funcs: vec![],
  labels: vec![],
});
