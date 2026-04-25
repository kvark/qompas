//! Grover's search algorithm — amplitude amplification.
//!
//! Operates directly on a [`StateVec`] for maximum efficiency. The algorithm
//! finds a marked state in an unstructured database of 2^n items using only
//! O(√N) queries.
//!
//! # Example
//!
//! ```rust
//! use qompas::state::StateVec;
//! use qompas::algorithms;
//!
//! let n = 4;
//! let target = 11; // search for |1011⟩
//! let iters = algorithms::optimal_iterations(n);
//!
//! let mut state = StateVec::new(n);
//! algorithms::prepare(&mut state);           // uniform superposition
//! for _ in 0..iters {
//!     algorithms::iterate(&mut state, target); // oracle + diffusion
//! }
//! let probs = state.probabilities();
//! assert!(probs[target] > 0.9);
//! ```

use crate::state::StateVec;
use num_complex::Complex64;

/// Optimal number of Grover iterations for `n` qubits.
///
/// This is ⌊π/4 · √N⌋ where N = 2^n.
pub fn optimal_iterations(num_qubits: usize) -> usize {
    let n = (1u64 << num_qubits) as f64;
    (std::f64::consts::FRAC_PI_4 * n.sqrt()).floor() as usize
}

/// Prepare the uniform superposition |s⟩ = H⊗n |0⟩.
///
/// After this, every basis state has amplitude 1/√N.
pub fn prepare(state: &mut StateVec) {
    let dim = state.amplitudes.len();
    let amp = 1.0 / (dim as f64).sqrt();
    for a in state.amplitudes.iter_mut() {
        *a = Complex64::new(amp, 0.0);
    }
}

/// One Grover iteration: oracle + diffusion.
///
/// The oracle flips the phase of the marked state `target`.
/// The diffusion operator reflects about the uniform superposition.
pub fn iterate(state: &mut StateVec, target: usize) {
    oracle(state, target);
    diffusion(state);
}

/// Oracle: flip the sign of amplitude `target`.
///
/// This is the "black box" query that marks the searched item.
pub fn oracle(state: &mut StateVec, target: usize) {
    state.amplitudes[target] = -state.amplitudes[target];
}

/// Diffusion operator: reflect about the mean amplitude.
///
/// Implements 2|s⟩⟨s| − I where |s⟩ is the uniform superposition.
/// This amplifies the amplitude of the marked state.
pub fn diffusion(state: &mut StateVec) {
    let dim = state.amplitudes.len();
    let mean: Complex64 = state.amplitudes.iter().copied().sum::<Complex64>() / dim as f64;
    for a in state.amplitudes.iter_mut() {
        *a = 2.0 * mean - *a;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimal_iterations_4_qubits() {
        // N=16, π/4 · √16 = π ≈ 3.14, floor = 3
        assert_eq!(optimal_iterations(4), 3);
    }

    #[test]
    fn optimal_iterations_5_qubits() {
        // N=32, π/4 · √32 ≈ 4.44, floor = 4
        assert_eq!(optimal_iterations(5), 4);
    }

    #[test]
    fn grover_4_qubits_finds_target() {
        let n = 4;
        let target = 11;
        let iters = optimal_iterations(n);

        let mut state = StateVec::new(n);
        prepare(&mut state);
        for _ in 0..iters {
            iterate(&mut state, target);
        }

        let probs = state.probabilities();
        assert!(
            probs[target] > 0.9,
            "target probability {:.4} should be > 0.9",
            probs[target]
        );
    }

    #[test]
    fn grover_5_qubits_finds_target() {
        let n = 5;
        let target = 19; // |10011⟩
        let iters = optimal_iterations(n);

        let mut state = StateVec::new(n);
        prepare(&mut state);
        for _ in 0..iters {
            iterate(&mut state, target);
        }

        let probs = state.probabilities();
        assert!(
            probs[target] > 0.99,
            "target probability {:.4} should be > 0.99",
            probs[target]
        );
    }

    #[test]
    fn over_rotation_decreases_probability() {
        let n = 4;
        let target = 7;
        let iters = optimal_iterations(n);

        let mut state = StateVec::new(n);
        prepare(&mut state);
        for _ in 0..iters {
            iterate(&mut state, target);
        }
        let prob_optimal = state.probabilities()[target];

        // One more iteration should decrease the probability
        iterate(&mut state, target);
        let prob_over = state.probabilities()[target];

        assert!(
            prob_over < prob_optimal,
            "over-rotation should decrease: {:.4} < {:.4}",
            prob_over,
            prob_optimal
        );
    }
}
