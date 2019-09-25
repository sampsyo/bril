#[macro_use]
extern crate clap;

#[macro_use]
extern crate log;
extern crate simplelog;

extern crate twox_hash;

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

  brilirs::run_input(input, std::io::stdout());
}
