extern crate strum;
#[macro_use]
extern crate strum_macros;

mod search;

fn main() {
    let prog_json = search::program::read_json().unwrap();
    let instructions = prog_json.functions[0].instrs.clone();
    let perturbed = search::mc_step(instructions.as_slice());
    println!("{:#?}", perturbed);
}