use clap::Parser;
use std::fs::File;

#[derive(Parser)]
#[clap(about, version, author)] // keeps the cli synced with Cargo.toml
struct Cli {
  /// Flag to output the total number of dynamic instructions
  #[clap(short, long)]
  profile: bool,

  /// The Bril file to run. stdin is assumed if file is not provided
  #[clap(short, long)]
  file: Option<String>,

  /// Flag to only typecheck/validate the bril program
  #[clap(short, long)]
  check: bool,

  /// Arguments for the main function
  args: Vec<String>,
}

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
  ) {
    eprintln!("error: {}", e);
    std::process::exit(2)
  }
}
