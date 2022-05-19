use bril_rs as bril;
use cranelift::frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift::codegen::{ir, isa};

fn tr_type(typ: &bril::Type) -> ir::Type {
    match typ {
        bril::Type::Int => ir::types::I32,
        bril::Type::Bool => ir::types::B1,
    }
}

fn tr_sig(func: &bril::Function) -> ir::Signature {
    let mut sig = ir::Signature::new(isa::CallConv::SystemV);
    if let Some(ret) = &func.return_type {
        sig.returns.push(ir::AbiParam::new(tr_type(ret)));
    }
    for arg in &func.args {
        sig.params.push(ir::AbiParam::new(tr_type(&arg.arg_type)));
    }
    sig
}

fn compile_func(func: bril::Function) {
    // Build function signature.
    let sig = tr_sig(&func);
    dbg!(sig);

    for inst in func.instrs {
        print!("{}\n", inst);
    }
}

fn main() {
    // Load the Bril program from stdin.
    let prog = bril::load_program();
    
    // Cranelift builder context.

    for func in prog.functions {
        compile_func(func);
    }
}
