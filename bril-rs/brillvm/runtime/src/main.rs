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
        if f.is_sign_negative() {
            print!("-Infinity");
        } else {
            print!("Infinity");
        }
    } else if f.is_nan() {
        print!("NaN");
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

#[cfg(not(test))]
#[panic_handler]
fn my_panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
