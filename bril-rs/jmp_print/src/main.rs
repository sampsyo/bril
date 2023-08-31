use bril_rs::{load_program_from_read, output_program};
use std::io;

fn main() -> io::Result<()> {
    let mut program = load_program_from_read(io::stdin());
    for func in &mut program.functions {
        let mut new_instrs = Vec::new();
        for code in &func.instrs {
            match code {
                bril_rs::Code::Instruction(bril_rs::Instruction::Effect {
                    op: bril_rs::EffectOps::Jump,
                    ..
                }) => {
                    new_instrs.push(bril_rs::Code::Instruction(bril_rs::Instruction::Constant {
                        dest: "tmp".to_string(),
                        op: bril_rs::ConstOps::Const,
                        const_type: bril_rs::Type::Int,
                        value: bril_rs::Literal::Int(69),
                    }));
                    new_instrs.push(bril_rs::Code::Instruction(bril_rs::Instruction::Effect {
                        args: vec!["tmp".to_string()],
                        funcs: Vec::new(),
                        labels: Vec::new(),
                        op: bril_rs::EffectOps::Print,
                    }));
                }
                _ => {}
            }
            new_instrs.push(code.clone());
        }
        func.instrs = new_instrs;
    }

    output_program(&program);
    Ok(())
}
