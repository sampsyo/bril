use std::alloc;
use std::mem::size_of;

#[unsafe(no_mangle)]
pub extern "C" fn print_int(i: i64) {
    print!("{i}");
}

#[unsafe(no_mangle)]
pub extern "C" fn print_bool(b: bool) {
    print!("{b}");
}

#[unsafe(no_mangle)]
pub extern "C" fn print_float(f: f64) {
    if f.is_infinite() {
        if f < 0.0 {
            print!("-Infinity");
        } else {
            print!("Infinity");
        }
    } else if f != 0.0 && f.abs().log10() >= 10.0 {
        print!("{}", format!("{f:.17e}").replace('e', "e+").as_str());
    } else if f != 0.0 && f.abs().log10() <= -10.0 {
        print!("{f:.17e}");
    } else {
        print!("{f:.17}");
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn print_sep() {
    print!(" ");
}

#[unsafe(no_mangle)]
pub extern "C" fn print_end() {
    println!();
}

const ALIGN: usize = 8;
const EXTRA_SIZE: usize = size_of::<usize>();

#[unsafe(no_mangle)]
pub extern "C" fn mem_alloc(count: i64, bytes: i64) -> *mut u8 {
    // The logical size of the allocation.
    let payload_size: usize = (count * bytes).try_into().unwrap();

    // Allocate one extra word to store the size.
    let layout = alloc::Layout::from_size_align(payload_size + EXTRA_SIZE, ALIGN).unwrap();

    unsafe {
        let ptr = alloc::alloc(layout);
        *(ptr as *mut usize) = payload_size;
        ptr.add(EXTRA_SIZE) // Pointer to the payload.
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mem_free(ptr: *mut u8) {
    // `ptr` points at the payload, which is immediately preceded by the size (which does not
    // include the size of the size itself).
    unsafe {
        let base_ptr = ptr.sub(EXTRA_SIZE);
        let payload_size = *(base_ptr as *mut usize);

        let layout = alloc::Layout::from_size_align(payload_size + EXTRA_SIZE, ALIGN).unwrap();
        alloc::dealloc(base_ptr, layout);
    }
}
