use bril_rs::load_program_from_read;
use brillvm::{cli::Cli, llvm::create_module_from_program};
use clap::Parser;
use inkwell::context::Context;
use std::io::Read;

fn main() {
    let args = Cli::parse();

    let mut src = String::new();
    if let Some(f) = &args.file {
        let path = std::fs::canonicalize(f).unwrap();
        let mut file = std::fs::File::open(path).unwrap();
        file.read_to_string(&mut src).unwrap()
    } else {
        std::io::stdin().read_to_string(&mut src).unwrap()
    };
    let prog = load_program_from_read(src.as_bytes());

    let context = Context::create();
    let llvm_prog = create_module_from_program(
        &context,
        &prog,
        args.runtime.as_ref().map_or("rt.bc", |f| f),
    );

    //println!("{}", prog);
    //llvm_prog.print_to_file("tmp.ll").unwrap();
    llvm_prog.verify().unwrap();
    /*     llvm_prog.print_to_stderr(); */
    println!("{}", llvm_prog.to_string());
}
