mod rt;
pub mod translator;

use crate::translator::{find_func, Translator};
use bril_rs::Program;
use cranelift_object::ObjectModule;

pub fn compile(program: Program, target: Option<String>, opt_level: &str, output: &str) {
    // Compile.
    let mut trans = Translator::<ObjectModule>::new(target, opt_level);
    trans.compile_prog(&program, false);

    // Add a C-style `main` wrapper.
    let main = find_func(&program.functions, "main");
    trans.add_c_main(&main.args, false);

    // Write object file.
    trans.emit(output);
}
