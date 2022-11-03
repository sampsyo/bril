use std::io::Read;

use bril_rs::output_program;

use rs2bril::cli::Cli;
use rs2bril::from_file_to_program;

use clap::Parser;

fn main() {
    let args = Cli::parse();
    let mut src = String::new();
    let source_name = if let Some(f) = args.file {
        let path = std::fs::canonicalize(f).unwrap();
        let mut file = std::fs::File::open(path.clone()).unwrap();
        file.read_to_string(&mut src).unwrap();
        Some(path.display().to_string())
    } else {
        std::io::stdin().read_to_string(&mut src).unwrap();
        None
    };

    let syntax = syn::parse_file(&src).unwrap();

    output_program(&from_file_to_program(syntax, args.position, source_name));
}
