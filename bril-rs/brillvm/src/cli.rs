use clap::Parser;

#[derive(Parser)]
#[command(about, version, author)] // keeps the cli synced with Cargo.toml
pub struct Cli {
    /// The bril file to be compiled to LLVM. stdin is assumed if file is not provided.
    #[arg(short, long, action)]
    pub file: Option<String>,

    #[arg(short, long, action)]
    pub runtime: Option<String>,
}
