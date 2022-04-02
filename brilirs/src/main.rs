use brilirs::cli::Cli;
use clap::Parser;
use std::fs::File;

fn main() {
  let args = Cli::parse();

  let input: Box<dyn std::io::Read> = match args.file {
    None => Box::new(std::io::stdin()),

    Some(input_file) => Box::new(File::open(input_file).unwrap()),
  };

  /*
  todo should you be able to supply output locations from the command line interface?
  Instead of builtin std::io::stdout()/std::io::stderr()
  */
  if let Err(e) = brilirs::run_input(
    input,
    std::io::stdout(),
    args.args,
    args.profile,
    std::io::stderr(),
    args.check,
    args.text,
  ) {
    eprintln!("error: {e}");
    std::process::exit(2)
  }
}
