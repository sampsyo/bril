use bril_rs as bril;
use brilift::translator::{find_func, Args, Translator};
use cranelift_jit::JITModule;
use cranelift_object::ObjectModule;

fn main() {
    let args: Args = argh::from_env();

    // Set up logging.
    simplelog::TermLogger::init(
        if args.verbose {
            simplelog::LevelFilter::Debug
        } else {
            simplelog::LevelFilter::Warn
        },
        simplelog::Config::default(),
        simplelog::TerminalMode::Mixed,
        simplelog::ColorChoice::Auto,
    )
    .unwrap();

    // Load the Bril program from stdin.
    let prog = bril::load_program();

    if args.jit {
        // Compile.
        let mut trans = Translator::<JITModule>::new();
        trans.compile_prog(&prog, args.dump_ir);

        // Add a JIT wrapper for `main`.
        let main = find_func(&prog.functions, "main");
        let entry_id = trans.add_mem_wrapper("main", &main.args, args.dump_ir);

        // Parse CLI arguments.
        if main.args.len() != args.args.len() {
            panic!(
                "@main expects {} arguments; got {}",
                main.args.len(),
                args.args.len()
            );
        }
        let main_args: Vec<bril::Literal> = main
            .args
            .iter()
            .zip(args.args)
            .map(|(arg, val_str)| match arg.arg_type {
                bril::Type::Int => bril::Literal::Int(val_str.parse().unwrap()),
                bril::Type::Bool => bril::Literal::Bool(val_str == "true"),
                bril::Type::Float => bril::Literal::Float(val_str.parse().unwrap()),
                bril::Type::Pointer(_) => unimplemented!("pointers not supported as main args"),
            })
            .collect();

        // Invoke the main function.
        unsafe { trans.run(entry_id, &main_args) };
    } else {
        // Compile.
        let mut trans = Translator::<ObjectModule>::new(args.target, &args.opt_level);
        trans.compile_prog(&prog, args.dump_ir);

        // Add a C-style `main` wrapper.
        let main = find_func(&prog.functions, "main");
        trans.add_c_main(&main.args, args.dump_ir);

        // Write object file.
        trans.emit(&args.output);
    }
}
