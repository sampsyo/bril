pub use interp::InterpError;

mod basic_block;
mod interp;

#[macro_use]
extern crate log;
extern crate simplelog;

extern crate bril_rs;
extern crate serde;
extern crate serde_derive;
extern crate serde_json;

pub fn run_input<T: std::io::Write>(input: Box<dyn std::io::Read>, out: T, input_args: Vec<&str>) {
  let prog = bril_rs::load_program_from_read(input);
  let bbprog = basic_block::BBProgram::new(prog);

  if let Err(e) = interp::execute_main(bbprog, out, input_args) {
    error!("{:?}", e);
    std::process::exit(2)
  };
}
