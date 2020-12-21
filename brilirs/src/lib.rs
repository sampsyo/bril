mod basic_block;
mod cfg;
mod interp;

#[macro_use]
extern crate log;
extern crate simplelog;

extern crate bril_rs;
extern crate serde;
extern crate serde_derive;
extern crate serde_json;

pub fn run_input<T: std::io::Write>(input: Box<dyn std::io::Read>, out: T) {
  let prog = bril_rs::load_program_from_read(input);
  let (main_idx, blocks, label_index) = basic_block::find_basic_blocks(prog);
  let blocks = cfg::build_cfg(blocks, &label_index);
  if let Err(e) = interp::execute((main_idx, blocks, label_index), out) {
    error!("{:?}", e);
  }
}
