#[no_mangle]
pub extern "C" fn print_int(i: i64) {
    print!("{}", i);
}

#[no_mangle]
pub extern "C" fn print_bool(b: bool) {
    print!("{}", b);
}

#[no_mangle]
pub extern "C" fn print_sep() {
    print!(" ");
}

#[no_mangle]
pub extern "C" fn print_end() {
    println!();
}
