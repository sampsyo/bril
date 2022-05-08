use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[clap(about, version, author)] // keeps the cli synced with Cargo.toml
pub struct Cli {
  pub path: PathBuf,
}
