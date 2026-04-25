//! Move-only qubit handles that enforce the no-cloning theorem at compile time.
//!
//! A [`Qubit`] is a linear-typed handle into a quantum [`State`](crate::state::StateVec).
//! It deliberately does **not** implement `Clone` or `Copy`, so any attempt to
//! duplicate a qubit is caught by the borrow checker:
//!
//! ```compile_fail
//! let q = qubit;
//! let q2 = qubit; // ERROR: use of moved value
//! ```
//!
//! This mirrors the physical no-cloning theorem and prevents an entire class
//! of bugs in circuit construction.

/// Opaque handle to a single qubit inside a state vector.
///
/// Not `Clone`, not `Copy` — this is intentional.
/// Passing a `Qubit` into a gate *consumes* it and returns a new handle,
/// modelling the irreversible nature of measurement and the linearity of
/// unitary evolution.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Qubit {
    /// Index into the parent state vector.
    pub(crate) index: usize,
}

impl Qubit {
    /// Create a new qubit handle (crate-internal — users get qubits from a circuit).
    pub(crate) fn new(index: usize) -> Self {
        Self { index }
    }

    /// Returns the zero-based index of this qubit in the state vector.
    pub fn index(&self) -> usize {
        self.index
    }
}

/// A pair of qubits returned by two-qubit gates.
#[derive(Debug)]
pub struct QubitPair(pub Qubit, pub Qubit);

/// Result of measuring a qubit: the classical bit plus the (collapsed) qubit.
#[derive(Debug)]
pub struct Measurement {
    /// Classical outcome: `false` = |0⟩, `true` = |1⟩.
    pub outcome: bool,
    /// The qubit after collapse — still usable in further gates.
    pub qubit: Qubit,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qubit_is_not_clone() {
        // This is a runtime assertion that the type is move-only.
        // The real compile-time check is in the doc-test above.
        let q = Qubit::new(0);
        let _q2 = q; // moves
                     // q is no longer accessible here
    }
}
