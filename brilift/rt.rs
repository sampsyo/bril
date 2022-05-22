#![no_std]

extern {
    pub fn printf(format: *const u8, ...) -> i32;
}

#[no_mangle]
pub extern "C" fn print_int(i: i64) {
    unsafe {
        printf("%ld\n\0".as_ptr() as *const _, i);
    }
}

#[panic_handler]
fn my_panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
