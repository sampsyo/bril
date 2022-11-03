use bril_rs::Position;
use brilirs::cli::Cli;
use brilirs::error::PositionalInterpError;
use clap::Parser;
use std::fs::File;
use std::io::Read;

fn main() {
  let args = Cli::parse();

  let input: Box<dyn std::io::Read> = match args.file.clone() {
    None => Box::new(std::io::stdin()),

    Some(input_file) => Box::new(File::open(input_file).unwrap()),
  };

  /*
  todo should you be able to supply output locations from the command line interface?
  Instead of builtin std::io::stdout()/std::io::stderr()
  */

  if let Err(e) = brilirs::run_input(
    input,
    std::io::BufWriter::new(std::io::stdout()),
    &args.args,
    args.profile,
    std::io::stderr(),
    args.check,
    args.text,
    args.file,
  ) {
    eprintln!("error: {e}");
    if let PositionalInterpError {
      pos: Some(Position {
        pos,
        pos_end,
        src: Some(src),
      }),
      ..
    } = e
    {
      let mut f = String::new();
      File::open(src).unwrap().read_to_string(&mut f).unwrap();

      let mut lines = f.split('\n');

      // print the first line
      eprintln!("{}", lines.nth((pos.row - 1) as usize).unwrap());
      eprintln!("{:>width$}", "^", width = pos.col as usize);

      // Then check if there is more
      if let Some(end) = pos_end {
        if pos.row != end.row {
          let mut row = pos.row + 1;
          while row < end.row {
            eprintln!("{}", lines.nth((row - 1) as usize).unwrap());
            eprintln!("^");
            row += 1;
          }
          eprintln!("{}", lines.nth((end.row - 1) as usize).unwrap());
          eprintln!("{:>width$}", "^", width = end.col as usize);
        }
      }
    }
    std::process::exit(2)
  }
}
