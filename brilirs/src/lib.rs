#![feature(or_patterns)]

use error::InterpError;

mod basic_block;
mod check;
mod error;
mod interp;

pub fn run_input<T: std::io::Write>(
    input: Box<dyn std::io::Read>,
    out: T,
    input_args: Vec<&str>,
    profiling: bool,
    check: bool,
) -> Result<(), InterpError> {
    let prog = bril_rs::load_program_from_read(input);
    let bbprog = basic_block::BBProgram::new(prog)?;
    check::type_check(&bbprog)?;

    if !check {
        interp::execute_main(bbprog, out, input_args, profiling)?;
    }

    Ok(())
}
