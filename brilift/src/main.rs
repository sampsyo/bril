use bril_rs::load_abstract_program;

fn main() {
    let prog = load_abstract_program();
    for func in prog.functions {
        for inst in func.instrs {
            print!("{}\n", inst);
        }
    }
}
