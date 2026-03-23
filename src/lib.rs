// Clippy rules borrowed from blade-graphics
#![allow(
    clippy::match_like_matches_macro,
    clippy::redundant_pattern_matching,
    clippy::needless_lifetimes,
    clippy::new_without_default,
    clippy::single_match,
    clippy::vec_init_then_push,
    clippy::missing_safety_doc
)]
#![warn(
    trivial_numeric_casts,
    unused_extern_crates,
    clippy::pattern_type_mismatch
)]

//! # Qompas — Quantum Exploration Library
//!
//! GPU-accelerated quantum circuit simulation and interactive visualization,
//! built on [blade-graphics](https://github.com/kvark/blade) and [egui](https://github.com/emilk/egui).
//!
//! ## Design Principles
//!
//! - **Borrow-checker–enforced quantum semantics**: The no-cloning theorem is
//!   mirrored by Rust's ownership model. Qubit handles are move-only types that
//!   prevent accidental duplication at compile time.
//! - **Portable HW acceleration**: blade-graphics provides a thin GPU abstraction
//!   over Vulkan, Metal, and DX12, giving us fast state-vector evolution on all
//!   desktop platforms.
//! - **Rich interactive visualization**: egui powers an immediate-mode UI for
//!   building, inspecting, and stepping through circuits in real time.
//! - **Standard format interop**: Parse and export circuits in OpenQASM and
//!   other established formats.

pub mod algorithms;
pub mod backend;
pub mod circuit;
pub mod gates;
pub mod qubit;
pub mod state;
pub mod state_gpu;
pub mod formats;
pub mod viz;
