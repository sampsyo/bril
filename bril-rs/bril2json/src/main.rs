use bril2json::cli::Cli;
use bril2json::parse_abstract_program;
use bril_rs::output_abstract_program;
use clap::Parser;

fn main() {
    let args = Cli::parse();
    output_abstract_program(&parse_abstract_program(
        args.position >= 1,
        args.position >= 2,
        args.file,
    ))
}
