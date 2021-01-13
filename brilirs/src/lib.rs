#![feature(or_patterns)]

mod basic_block;
mod check;
mod error;
mod interp;

pub fn run_input<T: std::io::Write>(
  input: Box<dyn std::io::Read>,
  out: T,
  input_args: Vec<&str>,
  profiling: bool,
) {
  let prog = bril_rs::load_program_from_read(input);
  let bbprog = basic_block::BBProgram::new(prog);

  if let Err(e) = check::type_check(&bbprog) {
    eprintln!("{:?}", e);
    std::process::exit(2)
  };

  if let Err(e) = interp::execute_main(bbprog, out, input_args, profiling) {
    eprintln!("{:?}", e);
    std::process::exit(2)
  };
}
