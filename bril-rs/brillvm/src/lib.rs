#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::needless_for_each)]
#![doc = include_str!("../README.md")]
// When you run with --all-targets, you also get --target=all which includes --target=redox. This pulls in redox_sys via a chain of deps through inkwell-parking_lot which uses an older version of bitflags 1.3.2. Given backwards compatibility, it's going to be a very long time, if ever, that this gets updated(because of msrv changes).
#![allow(clippy::multiple_crate_versions)]

#[doc(hidden)]
pub mod cli;

/// The Bril to LLVM IR compiler.
pub mod llvm;
