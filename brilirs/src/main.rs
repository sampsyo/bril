use brilirs::cli::Cli;
use brilirs::error::PositionalInterpError;
use clap::Parser;
use std::fs::File;
use std::io::Read;

fn main() {
  let args = Cli::parse();

  let mut input: Box<dyn std::io::Read> = match args.file {
    None => Box::new(std::io::stdin()),

    Some(input_file) => Box::new(File::open(input_file).unwrap()),
  };

  // Here we are reading out the input into a string so that we have it for error reporting
  // This will be done again during parsing inside of run_input because of the current interface
  // This is inefficient but probably not meaningfully so since benchmarking has shown that parsing quickly gets out weighted by program execution
  // If this does matter to you in your profiling, split up parse_abstract_program_from_read/load_abstract_program_from_read so that they can take a string instead of a `Read`
  let mut input_string = String::new();
  input.read_to_string(&mut input_string).unwrap();

  /*
  todo should you be able to supply output locations from the command line interface?
  Instead of builtin std::io::stdout()/std::io::stderr()
  */

  if let Err(e) = brilirs::run_input(
    input_string.as_bytes(),
    std::io::BufWriter::new(std::io::stdout()),
    &args.args,
    args.profile,
    std::io::stderr(),
    args.check,
    args.text,
  ) {
    let mut source_file = None;
    if args.text {
      source_file = Some(&input_string);
    }

    let mut tmp_string = String::new();
    if let Some(s) = args.source {
      File::open(s)
        .unwrap()
        .read_to_string(&mut tmp_string)
        .unwrap();
      source_file = Some(&tmp_string);
    }

    if let (
      Some(f),
      PositionalInterpError {
        e: _,
        pos: Some(pos),
      },
    ) = (source_file, &e)
    {
      // TODO delegate this to a crate that uses spans?
      let mut lines = f.split('\n');
      eprintln!("error: {e}");
      eprintln!("{}", lines.nth((pos.row - 1) as usize).unwrap());
      eprintln!("{:>width$}", "^", width = pos.col as usize);
    } else {
      eprintln!("error: {e}");
    }
    std::process::exit(2)
  }
}
