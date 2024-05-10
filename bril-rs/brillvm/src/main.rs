use brillvm::{cli::run, cli::Cli};
use clap::Parser;

fn main() {
    let args = Cli::parse();

    println!("{}", run(&args));
}
