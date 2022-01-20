use clap::{AppSettings, Parser};

#[derive(Parser)]
#[clap(about, version, author)] // keeps the cli synced with Cargo.toml
#[clap(setting(AppSettings::AllowHyphenValues))]
pub struct Cli {
  /// Flag to output the total number of dynamic instructions
  #[clap(short, long)]
  pub profile: bool,

  /// The bril file to run. stdin is assumed if file is not provided
  #[clap(short, long)]
  pub file: Option<String>,

  /// Flag to only typecheck/validate the bril program
  #[clap(short, long)]
  pub check: bool,

  /// Flag for when the bril program is in text form
  #[clap(short, long)]
  pub text: bool,

  /// Arguments for the main function
  pub args: Vec<String>,
}
