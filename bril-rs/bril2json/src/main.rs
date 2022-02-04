use bril2json::cli::Cli;
use bril2json::load_abstract_program;
use bril_rs::output_abstract_program;
use clap::Parser;

fn main() {
    let args = Cli::parse();
    output_abstract_program(&load_abstract_program(args.position))
}
