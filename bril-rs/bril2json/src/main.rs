use bril2json::load_abstract_program;
use bril_rs::output_abstract_program;

fn main() {
    output_abstract_program(&load_abstract_program())
}
