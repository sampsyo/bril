fn main() {
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-arg=-nostdlib");
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-arg=-undefined");
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-arg=dynamic_lookup");

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    assert!(false)
}
