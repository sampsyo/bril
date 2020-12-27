mod basic_block;
mod interp;

#[macro_use]
extern crate log;
extern crate simplelog;

extern crate bril_rs;
extern crate serde;
extern crate serde_derive;
extern crate serde_json;

pub fn run_input<T: std::io::Write>(input: Box<dyn std::io::Read>, mut out: T) {
  let prog = bril_rs::load_program_from_read(input);
  println!("{:?}", &prog);
  let bbprog = basic_block::BBProgram::new(prog);
  if let Err(e) = interp::execute(bbprog, &mut out) {
    error!("{:?}", e);
  }
}
