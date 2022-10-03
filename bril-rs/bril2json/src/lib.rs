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

#[doc(hidden)]
#[derive(Clone)]
pub struct Lines {
    use_pos: bool,
    with_end: bool,
    new_lines: Vec<usize>,
    src_name: Option<String>,
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
                    .map(|(i, j)| (i + 1, j))
                    .fold(
                        ColRow {
                            col: index as u64,
                            row: 1,
                        },
                        |current, (line_num, idx)| {
                            if *idx < index {
                                ColRow {
                                    row: (line_num + 1) as u64,
                                    col: (index - idx) as u64,
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
    let input: Box<dyn std::io::Read> = match file_name.clone() {
        Some(f) => Box::new(File::open(f).unwrap()),
        None => Box::new(std::io::stdin()),
    };

    parse_abstract_program_from_read(input, use_pos, with_end, file_name)
}
