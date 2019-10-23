extern crate core;
use core::arch::x86_64::*;
use std::ptr::write;
use std::ptr::read;

#[no_mangle] 
pub fn add(data_a: *const i32, data_b: *const i32, data_c: *mut i32) -> i32 {
    unsafe {
        let a = *data_a;
        let b = *data_b;
        let c = a + b;
        write(data_c, c);
        std::mem::forget(data_c);
        1
    }
}

#[no_mangle] 
pub fn vadd(data_a: *const i32, data_b: *const i32, data_c: *mut i32) -> i32 {
    unsafe {
        let mem_a = data_a as *const _;
        let mem_b = data_b as *const _;
        let mem_c = data_c as *mut _;
        let a = _mm_load_si128(mem_a);
        let b = _mm_load_si128(mem_b);
        let c = _mm_add_epi32(a, b);
        _mm_store_si128(mem_c, c);
        std::mem::forget(mem_c);
        1
    }
}

#[no_mangle] 
pub fn vsub(data_a: *const i32, data_b: *const i32, data_c: *mut i32) -> i32 {
    unsafe {
        let mem_a = data_a as *const _;
        let mem_b = data_b as *const _;
        let mem_c = data_c as *mut _;
        let a = _mm_load_si128(mem_a);
        let b = _mm_load_si128(mem_b);
        let c = _mm_sub_epi32(a, b);
        _mm_store_si128(mem_c, c);
        std::mem::forget(mem_c);
        1
    }
}

#[no_mangle] 
pub fn vmul(data_a: *const i32, data_b: *const i32, data_c: *mut i32) -> i32 {
    unsafe {
        let mem_a = data_a as *const _;
        let mem_b = data_b as *const _;
        let mem_c = data_c as *mut _;
        let a = _mm_load_si128(mem_a);
        let b = _mm_load_si128(mem_b);
        let c = _mm_mullo_epi16(a, b);
        _mm_store_si128(mem_c, c);
        std::mem::forget(mem_c);
        1
    }
}

#[no_mangle] 
pub fn vstore(data_a: *const i32, data_c: *mut i32) -> i32 {
    unsafe {
        let mem_a = data_a as *const _;
        let mem_c = data_c as *mut _;
        let a = _mm_load_si128(mem_a);
        _mm_store_si128(mem_c, a);
        std::mem::forget(mem_c);
        1
    }
}


