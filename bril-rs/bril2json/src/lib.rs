#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

// Tell the github workflow check to not format the generated rust program bril_grammar.rs
#[doc(hidden)]
#[rustfmt::skip]
pub mod bril_grammar;
#[doc(hidden)]
pub mod cli;
use std::fs::File;

use bril_rs::{AbstractProgram, ColRow, Position};

/// A helper function for processing the accepted Bril characters from their text representation
#[must_use]
pub fn escape_control_chars(s: &str) -> Option<char> {
    match s {
        "\\0" => Some('\u{0000}'),
        "\\a" => Some('\u{0007}'),
        "\\b" => Some('\u{0008}'),
        "\\t" => Some('\u{0009}'),
        "\\n" => Some('\u{000A}'),
        "\\v" => Some('\u{000B}'),
        "\\f" => Some('\u{000C}'),
        "\\r" => Some('\u{000D}'),
        s if s.chars().count() == 1 => s.chars().next(),
        _ => None,
    }
}

#[doc(hidden)]
#[derive(Clone)]
pub struct Lines {
    use_pos: bool,
    with_end: bool,
    new_lines: Vec<usize>,
    src_name: Option<String>,
}

// For use in the parser
enum ParsingArgs {
    Func(String),
    Ident(String),
    Label(String),
}

impl Lines {
    fn new(input: &str, use_pos: bool, with_end: bool, src_name: Option<String>) -> Self {
        Self {
            use_pos,
            with_end,
            src_name,
            new_lines: input
                .as_bytes()
                .iter()
                .enumerate()
                .filter_map(|(idx, b)| if *b == b'\n' { Some(idx) } else { None })
                .collect(),
        }
    }

    fn get_position(&self, starting_index: usize, ending_index: usize) -> Option<Position> {
        if self.use_pos {
            Some(Position {
                pos: self.get_row_col(starting_index).unwrap(),
                pos_end: if self.with_end {
                    self.get_row_col(ending_index)
                } else {
                    None
                },
                src: self.src_name.clone(),
            })
        } else {
            None
        }
    }

    fn get_row_col(&self, index: usize) -> Option<ColRow> {
        if self.use_pos {
            Some(
                self.new_lines
                    .iter()
                    .enumerate()
                    //(i+1) because line numbers start at 1
                    .map(|(i, j)| (i + 1, j))
                    .fold(
                        ColRow {
                            // (index + 1) because column numbers start at 1
                            col: (index + 1) as u64,
                            // Hard code the first row to be 1
                            row: 1,
                        },
                        |current, (line_num, idx)| {
                            if *idx < index {
                                ColRow {
                                    // (line_num + 1) because line numbers start at 1
                                    row: (line_num + 1) as u64,
                                    // column values are kept relative to the previous index
                                    col: ((index) - idx) as u64,
                                }
                            } else {
                                current
                            }
                        },
                    ),
            )
        } else {
            None
        }
    }
}

/// The entrance point to the bril2json parser. It takes an ```input```:[`std::io::Read`] which should be the Bril text file. You can control whether it includes source code positions with ```use_pos```.
/// # Panics
/// Will panic if the input is not well-formed Bril text
pub fn parse_abstract_program_from_read<R: std::io::Read>(
    mut input: R,
    use_pos: bool,
    with_end: bool,
    file_name: Option<String>,
) -> AbstractProgram {
    let mut buffer = String::new();
    input.read_to_string(&mut buffer).unwrap();
    let parser = bril_grammar::AbstractProgramParser::new();

    let src_name = file_name.map(|f| std::fs::canonicalize(f).unwrap().display().to_string());

    parser
        .parse(&Lines::new(&buffer, use_pos, with_end, src_name), &buffer)
        .unwrap()
}

#[must_use]
/// A wrapper around [`parse_abstract_program_from_read`] which assumes [`std::io::Stdin`] if `file_name` is [`None`]
/// # Panics
/// Will panic if the input is not well-formed Bril text or if `file_name` does not exist
pub fn parse_abstract_program(
    use_pos: bool,
    with_end: bool,
    file_name: Option<String>,
) -> AbstractProgram {
    let input = file_name.clone().map_or_else(
        || -> Box<dyn std::io::Read> { Box::new(std::io::stdin()) },
        |f| Box::new(File::open(f).unwrap()),
    );

    parse_abstract_program_from_read(input, use_pos, with_end, file_name)
}
