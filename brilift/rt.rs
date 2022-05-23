#![no_std]

extern {
    pub fn printf(format: *const u8, ...) -> i32;
    pub fn sscanf(s: *const u8, format: *const u8, ...) -> i32;
}

#[no_mangle]
pub extern "C" fn print_int(i: i64) {
    unsafe {
        printf("%ld\n\0".as_ptr() as *const _, i);
    }
}

#[no_mangle]
pub extern "C" fn parse_int(argv: *const *const u8, argi: i64) -> i64 {
    unsafe {
        let mut res: i64 = 0;
        sscanf(*argv.offset(argi.try_into().unwrap()), "%ld\0".as_ptr() as *const _, &mut res as *mut i64);
        res
    }
}

#[panic_handler]
fn my_panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
