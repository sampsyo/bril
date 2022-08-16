use clap::Parser;

#[derive(Parser)]
#[clap(about, version, author)] // keeps the cli synced with Cargo.toml
pub struct Cli {
    /// The bril file to statically link. stdin is assumed if file is not provided.
    #[clap(short, long, action)]
    pub file: Option<String>,
    /// Flag for whether position information should be included
    #[clap(short, action)]
    pub position: bool,
}
