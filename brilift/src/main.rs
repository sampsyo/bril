use argh::FromArgs;
use bril_rs as bril;
use brilift::compile;
use brilift::translator::{find_func, Translator};
use cranelift_jit::JITModule;
use std::str::FromStr;

#[derive(FromArgs)]
#[argh(description = "Bril compiler")]
struct RunArgs {
    #[argh(switch, short = 'j', description = "JIT and run (doesn't work)")]
    jit: bool,

    #[argh(option, short = 't', description = "target triple")]
    target: Option<String>,

    #[argh(
        option,
        short = 'o',
        description = "output object file",
        default = "String::from(\"bril.o\")"
    )]
    output: String,

    #[argh(switch, short = 'd', description = "dump CLIF IR")]
    dump_ir: bool,

    #[argh(switch, short = 'v', description = "verbose logging")]
    verbose: bool,

    #[argh(
        option,
        short = 'O',
        description = "optimization level (none, speed, or speed_and_size)",
        default = "OptLevel::None"
    )]
    opt_level: OptLevel,

    #[argh(
        positional,
        description = "arguments for @main function (JIT mode only)"
    )]
    args: Vec<String>,
}

pub enum OptLevel {
    None,
    Speed,
    SpeedAndSize,
}

impl OptLevel {
    pub fn to_str(self) -> &'static str {
        match self {
            OptLevel::None => "none",
            OptLevel::Speed => "speed",
            OptLevel::SpeedAndSize => "speed_and_size",
        }
    }
}

impl FromStr for OptLevel {
    type Err = String;
    fn from_str(s: &str) -> Result<OptLevel, String> {
        match s {
            "none" => Ok(OptLevel::None),
            "speed" => Ok(OptLevel::Speed),
            "speed_and_size" => Ok(OptLevel::SpeedAndSize),
            _ => Err(format!("unknown optimization level {s}")),
        }
    }
}

fn main() {
    let args: RunArgs = argh::from_env();

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
                bril::Type::Char => bril::Literal::Char(val_str.parse().unwrap()),
                bril::Type::Pointer(_) => unimplemented!("pointers not supported as main args"),
            })
            .collect();

        // Invoke the main function.
        unsafe { trans.run(entry_id, &main_args) };
    } else {
        compile(
            &prog,
            args.target.clone(),
            &args.output,
            args.opt_level.to_str(),
            args.dump_ir,
        );
    }
}
