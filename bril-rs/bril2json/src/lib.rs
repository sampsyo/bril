#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

// Tell the github workflow check to not format the generated rust program bril_grammar.rs
#[doc(hidden)]
#[rustfmt::skip]
pub mod bril_grammar;
#[doc(hidden)]
pub mod cli;
use bril_rs::{AbstractProgram, Position};

#[doc(hidden)]
#[derive(Clone)]
pub struct Lines {
    use_pos: bool,
    new_lines: Vec<usize>,
}

impl Lines {
    fn new(input: &str, use_pos: bool) -> Self {
        Self {
            use_pos,
            new_lines: input
                .as_bytes()
                .iter()
                .enumerate()
                .filter_map(|(idx, b)| if *b == b'\n' { Some(idx) } else { None })
                .collect(),
        }
    }

    fn get_position(&self, index: usize) -> Option<Position> {
        if self.use_pos {
            Some(
                self.new_lines
                    .iter()
                    .enumerate()
                    .map(|(i, j)| (i + 1, j))
                    .fold(
                        Position {
                            col: index as u64,
                            row: 1,
                        },
                        |current, (line_num, idx)| {
                            if *idx < index {
                                Position {
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
) -> AbstractProgram {
    let mut buffer = String::new();
    input.read_to_string(&mut buffer).unwrap();
    let parser = bril_grammar::AbstractProgramParser::new();
    parser
        .parse(&Lines::new(&buffer, use_pos), &buffer)
        .unwrap()
}

#[must_use]
/// A wrapper around [`parse_abstract_program_from_read`] which assumes [`std::io::Stdin`]
pub fn parse_abstract_program(use_pos: bool) -> AbstractProgram {
    parse_abstract_program_from_read(std::io::stdin(), use_pos)
}
