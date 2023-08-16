use bril_rs::load_program_from_read;
use brillvm::{cli::Cli, llvm::create_module_from_program};
use clap::Parser;
use inkwell::{
    context::Context,
    module::Module,
    targets::{InitializationConfig, Target},
};
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
    let runtime_path = args.runtime.as_ref().map_or("rt.bc", |f| f);
    // create a module from the runtime library for functions like printing/parsing
    let runtime_module = Module::parse_bitcode_from_path(runtime_path, &context).unwrap();
    let llvm_prog = create_module_from_program(&context, &prog, runtime_module);

    //println!("{}", prog);
    //llvm_prog.print_to_file("tmp.ll").unwrap();
    llvm_prog.verify().unwrap();

    if args.interpreter {
        Target::initialize_native(&InitializationConfig::default())
            .expect("Failed to initialize native target");

        let engine = llvm_prog
            .create_jit_execution_engine(inkwell::OptimizationLevel::None)
            .unwrap();

        let mut args: Vec<&str> = args.args.iter().map(|s| s.as_ref()).collect();
        args.insert(0, "bril_prog");
        unsafe {
            engine.run_function_as_main(llvm_prog.get_function("main").unwrap(), &args);
        }
    } else {
        println!("{}", llvm_prog.to_string())
    }
}
