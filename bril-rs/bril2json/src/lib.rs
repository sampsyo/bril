#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
// todo these are allowed to appease clippy but should be addressed some day
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::cargo_common_metadata)]

pub mod bril_grammar;
pub mod cli;
use bril_rs::{AbstractProgram, Position};

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
            Some(self.new_lines.iter().enumerate().fold(
                Position { col: 1, row: 0 },
                |current, (line_num, idx)| {
                    if *idx < index {
                        Position {
                            row: (line_num + 2) as u64,
                            col: (index - idx) as u64,
                        }
                    } else {
                        current
                    }
                },
            ))
        } else {
            None
        }
    }
}

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

pub fn parse_abstract_program(use_pos: bool) -> AbstractProgram {
    parse_abstract_program_from_read(std::io::stdin(), use_pos)
}