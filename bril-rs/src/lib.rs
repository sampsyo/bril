#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
// todo these are allowed to appease clippy but should be addressed some day
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::cargo_common_metadata)]

pub mod conversion;
pub mod program;
pub use conversion::ConversionError;
pub use program::*;

pub mod abstract_program;
pub use abstract_program::*;

use std::io::{self, Write};

pub fn load_program_from_read<R: std::io::Read>(mut input: R) -> Program {
    let mut buffer = String::new();
    input.read_to_string(&mut buffer).unwrap();
    serde_json::from_str(&buffer).unwrap()
}

pub fn load_program() -> Program {
    load_program_from_read(std::io::stdin())
}

pub fn output_program(p: &Program) {
    serde_json::to_writer_pretty(io::stdout(), p).unwrap();
    io::stdout().write_all(b"\n").unwrap();
}

pub fn load_abstract_program_from_read<R: std::io::Read>(mut input: R) -> AbstractProgram {
    let mut buffer = String::new();
    input.read_to_string(&mut buffer).unwrap();
    serde_json::from_str(&buffer).unwrap()
}

pub fn load_abstract_program() -> AbstractProgram {
    load_abstract_program_from_read(std::io::stdin())
}

pub fn output_abstract_program(p: &AbstractProgram) {
    serde_json::to_writer_pretty(io::stdout(), p).unwrap();
    io::stdout().write_all(b"\n").unwrap();
}
