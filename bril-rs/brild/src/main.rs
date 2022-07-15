use clap::Parser;

use std::collections::HashMap;
use std::fs::canonicalize;
use std::path::PathBuf;

use bril_rs::{load_abstract_program, output_abstract_program, AbstractProgram};
use brild::{cli::Cli, do_import, error::BrildError, handle_program};

fn main() -> Result<(), BrildError> {
    let mut map = HashMap::new();
    let args = Cli::parse();

    if let Some(p) = args.file {
        let path = PathBuf::from(p);
        do_import(&mut map, &canonicalize(path)?, &args.libs, true)?;
    } else {
        let program = load_abstract_program();
        // Note, since there is no path here, if you have a different file also importing this file it will import the same code twice with different names.
        handle_program(&mut map, program, &PathBuf::new(), &args.libs, true)?;
    }

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
