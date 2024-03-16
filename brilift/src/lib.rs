mod rt;
pub mod translator;

use crate::translator::{find_func, Translator};
use bril_rs::Program;
use cranelift_object::ObjectModule;
use std::str::FromStr;

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

pub struct CompileArgs<'a> {
    pub program: &'a Program,
    pub target: Option<String>,
    pub output: &'a str,
    pub opt_level: OptLevel,
    pub dump_ir: bool,
}

pub fn compile(args: CompileArgs) {
    // Compile.
    let mut trans = Translator::<ObjectModule>::new(args.target, args.opt_level.to_str());
    trans.compile_prog(args.program, args.dump_ir);

    // Add a C-style `main` wrapper.
    let main = find_func(&args.program.functions, "main");
    trans.add_c_main(&main.args, args.dump_ir);

    // Write object file.
    trans.emit(args.output);
}
