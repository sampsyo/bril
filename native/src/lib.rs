extern crate core;
use core::arch::x86_64::*;

#[no_mangle]
pub fn add(first: i32, second:i32) -> i32 {
 first + second
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


