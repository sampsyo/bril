#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]
#![allow(clippy::too_many_lines)]

/// Provides the unstructured representation of Bril programs
pub mod abstract_program;
/// Provides the Error handling and conversion between [`AbstractProgram`] and [Program]
pub mod conversion;
/// Provides the structured representation of Bril programs
pub mod program;
pub use abstract_program::*;
pub use program::*;

use std::io::{self, Write};

// todo Have versions of the output_* functions that take a [std::io::Write]
// todo possible deprecate/remove the wrapper functions to make the code base cleaner
// todo Wrap the outputs of serde in an error instead of panic-ing

/// A helper function for parsing a Bril program from ```input``` in JSON format to [Program]
/// # Panics
/// Will panic if the input JSON is not well-formed bril JSON
pub fn load_program_from_read<R: std::io::Read>(mut input: R) -> Program {
    let mut buffer = String::new();
    input.read_to_string(&mut buffer).unwrap();
    serde_json::from_str(&buffer).unwrap()
}

/// A wrapper of [`load_program_from_read`] which assumes [`std::io::Stdin`]
#[must_use]
pub fn load_program() -> Program {
    load_program_from_read(std::io::stdin())
}

/// Outputs a [Program] to [`std::io::Stdout`]
/// # Panics
/// This can panic, though I'm not sure when since serialization should always succeed
pub fn output_program(p: &Program) {
    serde_json::to_writer_pretty(io::stdout(), p).unwrap();
    io::stdout().write_all(b"\n").unwrap();
}

/// A helper function for parsing a Bril program from ```input``` in JSON format to [`AbstractProgram`]
/// # Panics
/// Will panic if the input JSON is not well-formed bril JSON
pub fn load_abstract_program_from_read<R: std::io::Read>(mut input: R) -> AbstractProgram {
    let mut buffer = String::new();
    input.read_to_string(&mut buffer).unwrap();
    serde_json::from_str(&buffer).unwrap()
}

/// A wrapper of [`load_abstract_program_from_read`] which assumes [`std::io::Stdin`]
#[must_use]
pub fn load_abstract_program() -> AbstractProgram {
    load_abstract_program_from_read(std::io::stdin())
}

/// Outputs an [`AbstractProgram`] to [`std::io::Stdout`]
/// # Panics
/// This can panic, though I'm not sure when since serialization should always succeed
pub fn output_abstract_program(p: &AbstractProgram) {
    serde_json::to_writer_pretty(io::stdout(), p).unwrap();
    io::stdout().write_all(b"\n").unwrap();
}
