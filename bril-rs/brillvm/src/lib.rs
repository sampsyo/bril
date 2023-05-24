#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::needless_for_each)]
#![doc = include_str!("../README.md")]

#[doc(hidden)]
pub mod cli;

/// The Bril to LLVM IR compiler.
pub mod llvm;
