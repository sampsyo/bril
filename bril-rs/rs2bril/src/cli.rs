use clap::Parser;

#[derive(Parser)]
#[clap(about, version, author)] // keeps the cli synced with Cargo.toml
pub struct Cli {
    /// Flag for whether position information should be included
    #[clap(short, action)]
    pub position: bool,
}
