use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(about, version, author)] // keeps the cli synced with Cargo.toml
pub struct Cli {
    /// The bril file to statically link. stdin is assumed if file is not provided.
    #[arg(short, long, action)]
    pub file: Option<String>,
    /// A list of library paths to look for Bril files.
    #[arg(short, long, action, num_args=1..)]
    pub libs: Vec<PathBuf>,
}
