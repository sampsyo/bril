#[no_mangle]
pub extern "C" fn print_int(i: i64) {
    print!("{}", i);
}

#[no_mangle]
pub extern "C" fn print_bool(b: bool) {
    print!("{}", b);
}

#[no_mangle]
pub extern "C" fn print_float(f: f64) {
    if f.is_infinite() {
        if f < 0.0 {
            print!("-Infinity");
        } else {
            print!("Infinity");
        }
    } else {
        print!("{}", f);
    }
}

#[no_mangle]
pub extern "C" fn print_sep() {
    print!(" ");
}

#[no_mangle]
pub extern "C" fn print_end() {
    println!();
}
