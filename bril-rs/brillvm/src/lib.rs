#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![warn(missing_docs)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::needless_for_each)]
#![doc = include_str!("../README.md")]

#[doc(hidden)]
pub mod cli;

/// The Bril to LLVM IR compiler.
pub mod llvm;
