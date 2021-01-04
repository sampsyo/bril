use clap::clap_app;
use std::fs::File;

fn main() {
  let args = clap_app!(brilirs =>
    (version: "0.1")
    (author: "Wil Thomason <wbthomason@cs.cornell.edu>")
    (about: "An interpreter for Bril")
    (@arg FILE: -f --file "The Bril file to run. stdin is assumed if FILE is not provided")
    (@arg args: ... "Arguments for the main function ")
  )
  .get_matches();

  let input_args = args.values_of("args").unwrap_or_default().collect();

  let input: Box<dyn std::io::Read> = match args.value_of("FILE") {
    None => {
      Box::new(std::io::stdin())
    }

    Some(input_file) => {
      Box::new(File::open(input_file).unwrap())
    }
  };

  brilirs::run_input(input, std::io::stdout(), input_args)
}
