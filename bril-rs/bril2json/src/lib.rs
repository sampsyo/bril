pub mod bril_grammar;
use bril_rs::AbstractProgram;

pub fn load_abstract_program_from_read<R: std::io::Read>(mut input: R) -> AbstractProgram {
    let mut buffer = String::new();
    input.read_to_string(&mut buffer).unwrap();
    let parser = bril_grammar::AbstractProgramParser::new();
    parser.parse(&buffer).unwrap()
}

pub fn load_abstract_program() -> AbstractProgram {
    load_abstract_program_from_read(std::io::stdin())
}
