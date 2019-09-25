mod basic_block;
mod cfg;
mod interp;
mod ir_types;
mod parse;

#[macro_use]
extern crate log;
extern crate simplelog;

#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate serde_json;

pub fn run_input<T: std::io::Write>(input: Box<dyn std::io::Read>, out: T) {
  match parse::load(input)
    .map(parse::convert_identifiers)
    .map(|(prog, num_vars)| (basic_block::find_basic_blocks(prog), num_vars - 1))
    .map(|((main_idx, blocks, label_index), num_vars)| {
      (
        main_idx,
        cfg::build_cfg(blocks, &label_index),
        label_index,
        num_vars,
      )
    }) {
    Ok((main_idx, blocks, label_index, num_vars)) => {
      dbg!(&main_idx);
      dbg!(&blocks);
      dbg!(&label_index);
      dbg!(num_vars);
      match interp::execute((main_idx, blocks, label_index), num_vars, out) {
        Ok(()) => (),
        Err(e) => error!("{:?}", e),
      }
    }
    Err(e) => error!("{}", e),
  }
}
