use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[clap(about, version, author)] // keeps the cli synced with Cargo.toml
pub struct Cli {
    /// The bril file to statically link. stdin is assumed if file is not provided.
    #[clap(short, long, action)]
    pub file: Option<String>,
    /// A list of library paths to look for Bril files.
    #[clap(short, long, action, multiple_values = true)]
    pub libs: Vec<PathBuf>,
}
