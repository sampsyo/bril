use clap::Parser;

#[derive(Parser)]
#[clap(about, version, author)] // keeps the cli synced with Cargo.toml
pub struct Cli {
  /// Flag to output the total number of dynamic instructions
  #[clap(short, long)]
  pub profile: bool,

  /// The Bril file to run. stdin is assumed if file is not provided
  #[clap(short, long)]
  pub file: Option<String>,

  /// Flag to only typecheck/validate the bril program
  #[clap(short, long)]
  pub check: bool,

  /// Arguments for the main function
  pub args: Vec<String>,
}
