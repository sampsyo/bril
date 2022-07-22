use std::io::Read;

use bril_rs::output_program;

use rs2bril::cli::Cli;
use rs2bril::from_file_to_program;

use clap::Parser;

fn main() {
    let args = Cli::parse();
    let mut src = String::new();
    std::io::stdin().read_to_string(&mut src).unwrap();
    let syntax = syn::parse_file(&src).unwrap();

    output_program(&from_file_to_program(syntax, args.position));
}
