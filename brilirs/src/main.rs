use brilirs::cli::Cli;
use clap::Parser;
use std::fs::File;

fn main() {
  let args = Cli::parse();

  let input: Box<dyn std::io::Read> = match args.file {
    None => Box::new(std::io::stdin()),

    Some(input_file) => Box::new(File::open(input_file).unwrap()),
  };

  if let Err(e) = brilirs::run_input(
    input,
    std::io::stdout(),
    args.args,
    args.profile,
    args.check,
    args.text,
  ) {
    eprintln!("error: {}", e);
    std::process::exit(2)
  }
}
