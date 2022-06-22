use clap::Parser;

#[derive(Parser)]
#[clap(about, version, author)] // keeps the cli synced with Cargo.toml
#[clap(allow_hyphen_values(true))]
pub struct Cli {
  /// Flag to output the total number of dynamic instructions
  #[clap(short, long, action)]
  pub profile: bool,

  /// The bril file to run. stdin is assumed if file is not provided
  #[clap(short, long, action)]
  pub file: Option<String>,

  /// Flag to only typecheck/validate the bril program
  #[clap(short, long, action)]
  pub check: bool,

  /// Flag for when the bril program is in text form
  #[clap(short, long, action)]
  pub text: bool,

  /// Arguments for the main function
  #[clap(action)]
  pub args: Vec<String>,

  /// This is the original source file that the file input was generated from.
  /// You would want to provide this if the bril file you are providing to the
  /// interpreter is in JSON form with source positions and you what brilirs to
  /// include sections of the source file in its error messages.
  /// If --text/-t is provided, that will be assumed to be the source file if none are provided.
  #[clap(short, long, action)]
  pub source: Option<String>,
}
