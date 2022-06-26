use std::alloc;

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

#[no_mangle]
pub extern "C" fn mem_alloc(count: i64, bytes: i64) -> *mut u8 {
    let size = count * bytes;
    let layout = alloc::Layout::from_size_align(size.try_into().unwrap(), bytes.try_into().unwrap()).unwrap();
    unsafe {
        return alloc::alloc(layout);
    }
}

#[no_mangle]
pub extern "C" fn mem_free(ptr: *mut u8) {
    // Nothing for now...
}
