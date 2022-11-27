/* use std::alloc::Layout;
use std::alloc::{alloc, dealloc};
use std::convert::TryInto;
use std::mem::size_of; */
#![no_std]
#![no_main]

use libc_print::{libc_print, libc_println};

#[no_mangle]
pub extern "C" fn print_int(i: i64) {
    libc_print!("{}", i);
}

#[no_mangle]
pub extern "C" fn print_bool(b: bool) {
    if b {
        libc_print!("true")
    } else {
        libc_print!("false")
    }
}

#[no_mangle]
pub extern "C" fn print_float(f: f64) {
    if f.is_infinite() {
        if f < 0.0 {
            libc_print!("-Infinity");
        } else {
            libc_print!("Infinity");
        }
    } else {
        libc_print!("{:.17}", f);
    }
}

#[no_mangle]
pub extern "C" fn print_sep() {
    libc_print!(" ");
}

#[no_mangle]
pub extern "C" fn print_end() {
    libc_println!();
}
/*
const ALIGN: usize = 8;
const EXTRA_SIZE: usize = size_of::<usize>();

#[no_mangle]
pub extern "C" fn mem_alloc(count: i64, bytes: i64) -> *mut u8 {
    // The logical size of the allocation.
    let payload_size: usize = (count * bytes).try_into().unwrap();

    // Allocate one extra word to store the size.
    let layout = Layout::from_size_align(payload_size + EXTRA_SIZE, ALIGN).unwrap();

    unsafe {
        let ptr = alloc(layout);
        *(ptr as *mut usize) = payload_size;
        ptr.add(EXTRA_SIZE) // Pointer to the payload.
    }
}

#[no_mangle]
pub fn mem_free(ptr: *mut u8) {
    // `ptr` points at the payload, which is immediately preceded by the size (which does not
    // include the size of the size itself).
    unsafe {
        let base_ptr = ptr.sub(EXTRA_SIZE);
        let payload_size = *(base_ptr as *mut usize);

        let layout =
            Layout::from_size_align(payload_size + EXTRA_SIZE, ALIGN).unwrap();
        dealloc(base_ptr, layout);
    }
} */

#[panic_handler]
fn my_panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
