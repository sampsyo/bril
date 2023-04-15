/* use std::alloc::Layout;
use std::alloc::{alloc, dealloc};
use std::convert::TryInto;
use std::mem::size_of; */
#![no_std]
#![no_main]

use core::ffi::{c_char, CStr};

use libc_print::std_name::{print, println};

#[no_mangle]
pub extern "C" fn _bril_print_int(i: i64) {
    print!("{}", i);
}

#[no_mangle]
pub extern "C" fn _bril_print_bool(b: bool) {
    if b {
        print!("true")
    } else {
        print!("false")
    }
}

#[no_mangle]
pub extern "C" fn _bril_print_float(f: f64) {
    if f.is_infinite() {
        if f < 0.0 {
            print!("-Infinity");
        } else {
            print!("Infinity");
        }
    } else {
        print!("{:.17}", f);
    }
}

#[no_mangle]
pub extern "C" fn _bril_print_sep() {
    print!(" ");
}

#[no_mangle]
pub extern "C" fn _bril_print_end() {
    println!();
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn _bril_parse_int(arg: *const c_char) -> i64 {
    let c_str = unsafe { CStr::from_ptr(arg) };
    let r_str = c_str.to_str().unwrap();
    r_str.parse::<i64>().unwrap()
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn _bril_parse_bool(arg: *const c_char) -> bool {
    let c_str = unsafe { CStr::from_ptr(arg) };
    let r_str = c_str.to_str().unwrap();
    r_str.parse::<bool>().unwrap()
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn _bril_parse_float(arg: *const c_char) -> f64 {
    let c_str = unsafe { CStr::from_ptr(arg) };
    let r_str = c_str.to_str().unwrap();
    r_str.parse::<f64>().unwrap()
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

#[cfg(not(test))]
#[panic_handler]
fn my_panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
