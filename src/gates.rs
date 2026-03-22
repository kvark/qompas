//! Standard quantum gates represented as small unitary matrices.
//!
//! Each gate function consumes its input [`Qubit`](crate::qubit::Qubit) handles
//! and returns new ones, preserving the move-only linearity invariant.

use nalgebra::Matrix2;
use num_complex::Complex64;
use std::f64::consts::{FRAC_1_SQRT_2, PI};

/// Convenience alias.
type C = Complex64;

fn c(re: f64, im: f64) -> C {
    C::new(re, im)
}

/// 2×2 unitary matrix representing a single-qubit gate.
#[derive(Debug, Clone)]
pub struct Gate1 {
    pub name: &'static str,
    pub matrix: Matrix2<C>,
}

/// 4×4 unitary matrix representing a two-qubit gate (stored as four 2×2 blocks).
#[derive(Debug, Clone)]
pub struct Gate2 {
    pub name: &'static str,
    /// Stored in row-major order as a flat [Complex64; 16].
    pub matrix: [C; 16],
}

// ── Single-qubit gates ──────────────────────────────────────────────

/// Pauli-X (NOT) gate.
pub fn x() -> Gate1 {
    Gate1 {
        name: "X",
        matrix: Matrix2::new(c(0., 0.), c(1., 0.), c(1., 0.), c(0., 0.)),
    }
}

/// Pauli-Y gate.
pub fn y() -> Gate1 {
    Gate1 {
        name: "Y",
        matrix: Matrix2::new(c(0., 0.), c(0., -1.), c(0., 1.), c(0., 0.)),
    }
}

/// Pauli-Z gate.
pub fn z() -> Gate1 {
    Gate1 {
        name: "Z",
        matrix: Matrix2::new(c(1., 0.), c(0., 0.), c(0., 0.), c(-1., 0.)),
    }
}

/// Hadamard gate.
pub fn h() -> Gate1 {
    let s = FRAC_1_SQRT_2;
    Gate1 {
        name: "H",
        matrix: Matrix2::new(c(s, 0.), c(s, 0.), c(s, 0.), c(-s, 0.)),
    }
}

/// S (phase) gate — √Z.
pub fn s() -> Gate1 {
    Gate1 {
        name: "S",
        matrix: Matrix2::new(c(1., 0.), c(0., 0.), c(0., 0.), c(0., 1.)),
    }
}

/// T gate — √S.
pub fn t() -> Gate1 {
    let t_phase = (PI / 4.0).cos();
    let t_sin = (PI / 4.0).sin();
    Gate1 {
        name: "T",
        matrix: Matrix2::new(c(1., 0.), c(0., 0.), c(0., 0.), c(t_phase, t_sin)),
    }
}

/// Rotation around the Z axis by `theta` radians.
pub fn rz(theta: f64) -> Gate1 {
    Gate1 {
        name: "Rz",
        matrix: Matrix2::new(
            c((-theta / 2.0).cos(), (-theta / 2.0).sin()),
            c(0., 0.),
            c(0., 0.),
            c((theta / 2.0).cos(), (theta / 2.0).sin()),
        ),
    }
}

// ── Two-qubit gates ─────────────────────────────────────────────────

/// CNOT (CX) gate.
pub fn cnot() -> Gate2 {
    let o = c(0., 0.);
    let i = c(1., 0.);
    #[rustfmt::skip]
    let matrix = [
        i, o, o, o,
        o, i, o, o,
        o, o, o, i,
        o, o, i, o,
    ];
    Gate2 {
        name: "CNOT",
        matrix,
    }
}

/// SWAP gate.
pub fn swap() -> Gate2 {
    let o = c(0., 0.);
    let i = c(1., 0.);
    #[rustfmt::skip]
    let matrix = [
        i, o, o, o,
        o, o, i, o,
        o, i, o, o,
        o, o, o, i,
    ];
    Gate2 {
        name: "SWAP",
        matrix,
    }
}

/// CZ (controlled-Z) gate.
pub fn cz() -> Gate2 {
    let o = c(0., 0.);
    let i = c(1., 0.);
    let m = c(-1., 0.);
    #[rustfmt::skip]
    let matrix = [
        i, o, o, o,
        o, i, o, o,
        o, o, i, o,
        o, o, o, m,
    ];
    Gate2 {
        name: "CZ",
        matrix,
    }
}
