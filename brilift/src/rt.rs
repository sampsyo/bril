use std::alloc;
use std::mem::size_of;

#[no_mangle]
pub extern "C" fn print_int(i: i64) {
    print!("{i}");
}

#[no_mangle]
pub extern "C" fn print_bool(b: bool) {
    print!("{b}");
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
        print!("{f:.17}");
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

const ALIGN: usize = 8;
const EXTRA_SIZE: usize = size_of::<usize>();

#[no_mangle]
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

#[no_mangle]
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
