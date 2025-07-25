use clap::Parser;

#[derive(Parser)]
#[command(about, version, author)] // keeps the CLI synced with Cargo.toml
#[command(allow_hyphen_values(true))]
pub struct Cli {
  /// Flag to output the total number of dynamic instructions
  #[arg(short, long, action)]
  pub profile: bool,

  /// The Bril file to run. Standard input is assumed if file is not provided
  #[arg(short, long, action)]
  pub file: Option<String>,

  /// Flag to only typecheck/validate the Bril program
  #[arg(short, long, action)]
  pub check: bool,

  /// Flag for when the Bril program is in text form
  #[arg(short, long, action)]
  pub text: bool,

  /// Arguments for the main function
  #[arg(action)]
  pub args: Vec<String>,
}
