use crate::ir_types::{Function, Instruction, Program};
use std::collections::HashMap;
use std::rc::Rc;

pub struct BasicBlock {
    instrs: Vec<Instruction>,
    exit: Vec<Rc<BasicBlock>>,
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
) -> (
    Option<Rc<BasicBlock>>,
    Vec<Rc<BasicBlock>>,
    HashMap<String, Rc<BasicBlock>>,
) {
    let mut main_fn = None;
    let mut blocks = Vec::new();
    let mut labels = HashMap::new();

    let mut bb_helper = |func: Function| -> Rc<BasicBlock> {
        let mut curr_block = BasicBlock::new();
        let root_block = None;
        let mut label = None;
        for instr in func.instrs.into_iter() {
            match instr {
                Instruction::Label(ref l) => {
                    if !curr_block.instrs.is_empty() {
                        blocks.push(Rc::new(curr_block));
                        if let Some(old_label) = label {
                            labels.insert(old_label, Rc::clone(blocks.last().unwrap()));
                        }

                        curr_block = BasicBlock::new();
                    }

                    label = Some(l.label.clone());
                    curr_block.instrs.push(instr);
                }

                Instruction::Jmp { .. } | Instruction::Br { .. } | Instruction::Ret { .. } => {
                    curr_block.instrs.push(instr);
                    blocks.push(Rc::new(curr_block));
                    if let Some(l) = label {
                        labels.insert(l, Rc::clone(blocks.last().unwrap()));
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
            blocks.push(Rc::new(curr_block));
            if let Some(l) = label {
                labels.insert(l, Rc::clone(blocks.last().unwrap()));
            }
        }

        root_block.unwrap()
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
