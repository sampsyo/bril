#[macro_use]
extern crate clap;

#[macro_use]
extern crate log;
extern crate simplelog;

#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate serde_json;

mod basic_block;
mod cfg;
mod ir_types;
mod parse;

use std::fs::File;

use simplelog::{Config, LevelFilter, TermLogger, TerminalMode};

fn main() {
  let args = clap_app!(brilirs =>
    (version: "0.1")
    (author: "Wil Thomason <wbthomason@cs.cornell.edu>")
    (about: "An interpreter for Bril")
    (@arg verbose: -v --verbose "Print debug information")
    (@arg FILE: "The Bril file to run. stdin is assumed if FILE is not provided")
  )
  .get_matches();

  // Super default log setup
  TermLogger::init(
    if args.is_present("verbose") {
      LevelFilter::Debug
    } else {
      LevelFilter::Info
    },
    Config::default(),
    TerminalMode::Mixed,
  )
  .unwrap();

  let input: Box<dyn std::io::Read> = match args.value_of("FILE") {
    None => {
      debug!("Reading from stdin");
      Box::new(std::io::stdin())
    }

    Some(input_file) => {
      debug!("Reading from {}", input_file);
      Box::new(File::open(input_file).unwrap())
    }
  };

  match parse::load(input).map(basic_block::find_basic_blocks).map(
    |(main_idx, blocks, label_index)| (main_idx, cfg::build_cfg(blocks, &label_index), label_index),
  ) {
    Ok((main_idx, blocks, label_index)) => {
      dbg!(main_idx);
      dbg!(blocks);
      dbg!(label_index);
    }
    Err(e) => error!("{}", e),
  }
}
