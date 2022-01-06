use clap::{clap_app, AppSettings};
use std::fs::File;

fn main() {
  let args = clap_app!(brilirs =>
    (version: "0.1")
    (author: "Wil Thomason <wbthomason@cs.cornell.edu>")
    (about: "An interpreter for Bril")
    (@arg profiling: -p "Flag to output the total number of dynamic instructions")
    (@arg FILE: -f --file +takes_value "The Bril file to run. stdin is assumed if FILE is not provided")
    (@arg check: --check "Flag to only typeckeck/validate the bril program")
    (@arg args: +allow_hyphen_values ... "Arguments for the main function")
  ).setting(AppSettings::AllowLeadingHyphen)
  .get_matches();

  let input_args = args.values_of("args").unwrap_or_default().collect();

  let input: Box<dyn std::io::Read> = match args.value_of("FILE") {
    None => Box::new(std::io::stdin()),

    Some(input_file) => Box::new(File::open(input_file).unwrap()),
  };

  if let Err(e) = brilirs::run_input(
    input,
    std::io::stdout(),
    input_args,
    args.is_present("profiling"),
    args.is_present("check"),
  ) {
    eprintln!("error: {}", e);
    std::process::exit(2)
  }
}
