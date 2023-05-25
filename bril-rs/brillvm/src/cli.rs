use clap::Parser;

#[derive(Parser)]
#[command(about, version, author)] // keeps the cli synced with Cargo.toml
#[command(allow_hyphen_values(true))] // allows for negative numbers
pub struct Cli {
    /// The bril file to be compiled to LLVM. stdin is assumed if file is not provided.
    #[arg(short, long, action)]
    pub file: Option<String>,

    /// The path to the runtime library. Defaults to rt.bc
    #[arg(short, long, action)]
    pub runtime: Option<String>,

    /// Whether to interpret the program instead of outputting LLVM
    #[arg(short, long, action)]
    pub interpreter: bool,

    /// Arguments for the main function
    #[arg(action)]
    pub args: Vec<String>,
}
