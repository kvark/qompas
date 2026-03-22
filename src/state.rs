//! Dense state-vector simulator for small-to-medium circuits.
//!
//! The [`StateVec`] stores 2^n complex amplitudes and applies gates via
//! direct matrix–vector products.  This is the CPU fallback; a future
//! `StateGpu` will off-load the heavy lifting to blade-graphics compute
//! shaders.

use crate::gates::{Gate1, Gate2};
use crate::qubit::{Measurement, Qubit, QubitPair};
use num_complex::Complex64;

/// Dense state vector of `n` qubits (2^n amplitudes).
pub struct StateVec {
    pub(crate) num_qubits: usize,
    pub(crate) amplitudes: Vec<Complex64>,
}

impl StateVec {
    /// Allocate a new state with `n` qubits, initialised to |00…0⟩.
    pub fn new(num_qubits: usize) -> Self {
        let len = 1 << num_qubits;
        let mut amps = vec![Complex64::new(0.0, 0.0); len];
        amps[0] = Complex64::new(1.0, 0.0);
        Self {
            num_qubits,
            amplitudes: amps,
        }
    }

    /// Number of qubits in this state.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Borrow the raw amplitude vector (length = 2^n).
    pub fn amplitudes(&self) -> &[Complex64] {
        &self.amplitudes
    }

    /// Allocate qubit handles for all qubits in this state.
    pub fn qubits(&self) -> Vec<Qubit> {
        (0..self.num_qubits).map(Qubit::new).collect()
    }

    /// Apply a single-qubit gate. Consumes the qubit and returns a new handle.
    pub fn apply1(&mut self, gate: &Gate1, q: Qubit) -> Qubit {
        let target = q.index;
        let n = self.amplitudes.len();
        let bit = 1 << target;

        for i in 0..n {
            if i & bit != 0 {
                continue;
            }
            let j = i | bit;
            let a0 = self.amplitudes[i];
            let a1 = self.amplitudes[j];
            self.amplitudes[i] = gate.matrix[(0, 0)] * a0 + gate.matrix[(0, 1)] * a1;
            self.amplitudes[j] = gate.matrix[(1, 0)] * a0 + gate.matrix[(1, 1)] * a1;
        }
        Qubit::new(target)
    }

    /// Apply a two-qubit gate. Consumes both qubits, returns a new pair.
    pub fn apply2(&mut self, gate: &Gate2, q0: Qubit, q1: Qubit) -> QubitPair {
        let t0 = q0.index;
        let t1 = q1.index;
        assert_ne!(t0, t1, "two-qubit gate requires distinct qubits");

        let n = self.amplitudes.len();
        let bit0 = 1 << t0;
        let bit1 = 1 << t1;

        for i in 0..n {
            // Only process the canonical index (both target bits = 0)
            if i & bit0 != 0 || i & bit1 != 0 {
                continue;
            }
            let i00 = i;
            let i01 = i | bit1;
            let i10 = i | bit0;
            let i11 = i | bit0 | bit1;

            let a = [
                self.amplitudes[i00],
                self.amplitudes[i01],
                self.amplitudes[i10],
                self.amplitudes[i11],
            ];

            for (row, idx) in [(0, i00), (1, i01), (2, i10), (3, i11)] {
                self.amplitudes[idx] = gate.matrix[row * 4]     * a[0]
                                     + gate.matrix[row * 4 + 1] * a[1]
                                     + gate.matrix[row * 4 + 2] * a[2]
                                     + gate.matrix[row * 4 + 3] * a[3];
            }
        }
        QubitPair(Qubit::new(t0), Qubit::new(t1))
    }

    /// Measure a qubit in the computational basis (Z-basis).
    ///
    /// Uses a deterministic "max-probability" rule (no RNG) so that
    /// simulations are reproducible.  A future version will accept an
    /// `&mut dyn Rng` for stochastic measurement.
    pub fn measure(&mut self, q: Qubit) -> Measurement {
        let target = q.index;
        let bit = 1 << target;
        let n = self.amplitudes.len();

        let mut prob0: f64 = 0.0;
        for i in 0..n {
            if i & bit == 0 {
                prob0 += self.amplitudes[i].norm_sqr();
            }
        }

        let outcome = prob0 < 0.5;
        let keep_bit_value = if outcome { bit } else { 0 };
        let norm = if outcome {
            (1.0 - prob0).sqrt()
        } else {
            prob0.sqrt()
        };

        for i in 0..n {
            if (i & bit) == keep_bit_value {
                self.amplitudes[i] /= norm;
            } else {
                self.amplitudes[i] = Complex64::new(0.0, 0.0);
            }
        }

        Measurement {
            outcome,
            qubit: Qubit::new(target),
        }
    }

    /// Probability of each computational-basis state.
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|a| a.norm_sqr()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates;

    #[test]
    fn initial_state_is_zero() {
        let st = StateVec::new(2);
        let probs = st.probabilities();
        assert!((probs[0] - 1.0).abs() < 1e-10);
        assert!(probs[1..].iter().all(|&p| p < 1e-10));
    }

    #[test]
    fn x_gate_flips() {
        let mut st = StateVec::new(1);
        let q = Qubit::new(0);
        let _q = st.apply1(&gates::x(), q);
        let probs = st.probabilities();
        assert!((probs[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn hadamard_creates_superposition() {
        let mut st = StateVec::new(1);
        let q = Qubit::new(0);
        let _q = st.apply1(&gates::h(), q);
        let probs = st.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn bell_state() {
        let mut st = StateVec::new(2);
        let qs = st.qubits();
        let mut qs = qs.into_iter();
        let q0 = qs.next().unwrap();
        let q1 = qs.next().unwrap();
        let q0 = st.apply1(&gates::h(), q0);
        let pair = st.apply2(&gates::cnot(), q0, q1);
        let _q0 = pair.0;
        let _q1 = pair.1;
        let probs = st.probabilities();
        // Bell state: |00⟩ + |11⟩ with equal probability
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[3] - 0.5).abs() < 1e-10);
    }
}
