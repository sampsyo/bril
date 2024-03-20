mod rt;
pub mod translator;

use crate::translator::{find_func, Translator};
use bril_rs::Program;
use cranelift_object::ObjectModule;

/// Compile a program ahead of time to an object file.
///
/// * `program` - the Bril program to compile
/// * `target` - the target triple, or None to target the host
/// * `output` - the filename where we should write the object file
/// * `opt_level` - a Cranelift optimization level
/// * `dump_ir` - optionally emit the Cranelift IR to stdout
pub fn compile(
    program: &Program,
    target: Option<String>,
    output: &str,
    opt_level: &str,
    dump_ir: bool,
) {
    // Compile.
    let mut trans = Translator::<ObjectModule>::new(target, opt_level);
    trans.compile_prog(program, dump_ir);

    // Add a C-style `main` wrapper.
    let main = find_func(&program.functions, "main");
    trans.add_c_main(&main.args, dump_ir);

    // Write object file.
    trans.emit(output);
}
