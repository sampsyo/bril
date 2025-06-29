#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![warn(missing_docs)]
#![warn(clippy::allow_attributes)]
#![allow(clippy::float_cmp)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_arguments)]
#![doc = include_str!("../README.md")]

use crate::cli::Cli;
use basic_block::BBProgram;
use bril_rs::Program;
use error::PositionalInterpError;

/// The internal representation of `brilirs`, provided a ```TryFrom<Program>``` conversion
pub mod basic_block;
/// Provides ```check::type_check``` to validate [Program]
pub mod check;
#[doc(hidden)]
pub mod cli;
#[doc(hidden)]
pub mod error;
/// Provides ```interp::execute_main``` to execute [Program] that have been converted into [`BBProgram`]
pub mod interp;
/// An optimized version of `bril_rs` with less indirection
pub mod ir;

#[doc(hidden)]
pub fn run_input<T: std::io::Write, U: std::io::Write>(
  input: impl std::io::Read,
  out: T,
  profiling_out: U,
  cli_args: Cli,
) -> Result<(), PositionalInterpError> {
  // It's a little confusing because of the naming conventions.
  //      - bril_rs takes file.json as input
  //      - bril2json takes file.bril as input
  let prog: Program = if cli_args.text {
    bril2json::parse_abstract_program_from_read(input, true, true, cli_args.file).try_into()?
  } else {
    bril_rs::load_abstract_program_from_read(input).try_into()?
  };
  check::type_check(&prog)?;
  let bbprog: BBProgram = prog.try_into()?;

  if !cli_args.check {
    interp::execute_main(
      &bbprog,
      out,
      &cli_args.args,
      cli_args.profile,
      profiling_out,
    )?;
  }

  Ok(())
}
