use clap::Parser;

use std::collections::HashMap;
use std::fs::canonicalize;

use bril_rs::{output_abstract_program, AbstractProgram};
use loadbril::{cli::Cli, locate_imports};

fn main() -> std::io::Result<()> {
    let mut map = HashMap::new();
    let args = Cli::parse();

    locate_imports(&mut map, &canonicalize(&args.path)?, true)?;

    let result = map.into_iter().fold(
        AbstractProgram {
            imports: Vec::new(),
            functions: Vec::new(),
        },
        |mut acc, (_, p)| {
            acc.functions.append(&mut p.unwrap().functions);
            acc
        },
    );

    output_abstract_program(&result);

    Ok(())
}
