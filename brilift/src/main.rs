use bril_rs as bril;
use cranelift::frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift::codegen::{ir, isa};

fn compile_type(typ: bril::Type) -> ir::Type {
    match typ {
        bril::Type::Int => ir::types::I32,
        bril::Type::Bool => ir::types::B1,
    }
}

fn compile_func(func: bril::Function) {
    // Build function signature.
    let mut sig = ir::Signature::new(isa::CallConv::SystemV);
    if let Some(ret) = func.return_type {
        sig.returns.push(ir::AbiParam::new(compile_type(ret)));
    }
    for arg in func.args {
        sig.params.push(ir::AbiParam::new(compile_type(arg.arg_type)));
    }
    
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
