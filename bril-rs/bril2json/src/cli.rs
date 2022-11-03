use clap::{ArgAction::Count, Parser};

#[derive(Parser)]
#[command(about, version, author)] // keeps the cli synced with Cargo.toml
pub struct Cli {
    /// The bril file to statically link. stdin is assumed if file is not provided.
    #[arg(short, long, action)]
    pub file: Option<String>,
    /// Flag for whether position information should be included
    #[arg(short, action = Count)]
    pub position: u8,
}
