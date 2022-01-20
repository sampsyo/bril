// The group clippy::pedantic is not used as it ends up being more annoying than useful
#![warn(clippy::all, clippy::nursery, clippy::cargo)]
// todo these are allowed to appease clippy but should be addressed some day
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::cargo_common_metadata)]
#![allow(clippy::too_many_arguments)]

use std::error::Error;

mod basic_block;
mod check;
pub mod cli;
mod error;
mod interp;

pub fn run_input<T: std::io::Write>(
  input: Box<dyn std::io::Read>,
  out: T,
  input_args: Vec<String>,
  profiling: bool,
  check: bool,
  text: bool,
) -> Result<(), Box<dyn Error>> {
  // It's a little confusing because of the naming conventions.
  //      - bril_rs takes file.json as input
  //      - bril2json takes file.bril as input
  let prog = if text {
    bril2json::load_abstract_program_from_read(input).try_into()?
  } else {
    bril_rs::load_program_from_read(input)
  };
  let bbprog = basic_block::BBProgram::new(prog)?;
  check::type_check(&bbprog)?;

  if !check {
    interp::execute_main(&bbprog, out, &input_args, profiling)?;
  }

  Ok(())
}
