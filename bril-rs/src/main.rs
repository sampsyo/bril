#[macro_use]
extern crate clap;

#[macro_use]
extern crate slog;
extern crate slog_term;

#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate serde_json;

mod basic_block;
mod cfg;
mod parse;
mod ir_types;

fn main() {
  let args = clap_app!(brili =>
    (version: "0.1")
    (author: "Wil Thomason <wbthomason@cs.cornell.edu>")
    (about: "An interpreter for Bril")
    (@arg verbose: -v --verbose "Print debug information")
    (@arg FILE: +required "The Bril file to run")
  )
  .get_matches();
}
