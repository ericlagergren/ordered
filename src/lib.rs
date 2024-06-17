//! TODO

//#![cfg_attr(not(any(test, feature = "std")), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

mod imp;

pub use imp::*;
