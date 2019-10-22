extern crate strum;
#[macro_use]
extern crate strum_macros;

mod program;

fn main() {
    let prog_json = program::read_json().unwrap();
    let mut instructions = &prog_json.functions[0].instrs;
    println!("{:#?}", instructions);
}