pub mod bril_grammar;
use bril_rs::Program;

pub fn load_program_from_read<R: std::io::Read>(mut input: R) -> Program {
    let mut buffer = String::new();
    input.read_to_string(&mut buffer).unwrap();
    let parser = bril_grammar::ProgramParser::new();
    parser.parse(&buffer).unwrap()
}

pub fn load_program() -> Program {
    load_program_from_read(std::io::stdin())
}
