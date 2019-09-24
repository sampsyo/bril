mod basic_block;
mod cfg;
mod ir_types;
mod parse;
mod interp;

#[macro_use]
extern crate log;
extern crate simplelog;

#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate serde_json;

pub fn run_input<T: std::io::Write>(input: Box<dyn std::io::Read>, out: T) {
  match parse::load(input).map(basic_block::find_basic_blocks).map(
    |(main_idx, blocks, label_index)| (main_idx, cfg::build_cfg(blocks, &label_index), label_index),
  ) {
    Ok((main_idx, blocks, label_index)) => {
      dbg!(&main_idx);
      dbg!(&blocks);
      dbg!(&label_index);
      match interp::execute((main_idx, blocks, label_index), out) {
        Ok(()) => (),
        Err(e) => println!("{:?}", e),
      }
    }
    Err(e) => error!("{}", e),
  }
}
