#[no_mangle]
pub extern "C" fn print_int(i: i64) {
    println!("{}", i);
}
