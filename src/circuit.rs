//! High-level circuit builder.
//!
//! [`Circuit`] records a sequence of gate operations and can be executed
//! against a [`StateVec`](crate::state::StateVec) in one shot, or stepped
//! through one gate at a time for interactive exploration.

use crate::gates::{self, Gate1, Gate2};
use crate::qubit::{Measurement, Qubit, QubitPair};
use crate::state::StateVec;

/// A recorded gate operation.
#[derive(Debug, Clone)]
pub enum Op {
    /// Single-qubit gate applied to qubit at `target`.
    Gate1 { gate: Gate1, target: usize },
    /// Two-qubit gate applied to qubits `control` and `target`.
    Gate2 {
        gate: Gate2,
        control: usize,
        target: usize,
    },
    /// Measurement on qubit `target`.
    Measure { target: usize },
    /// Grover oracle — flip phase of a marked basis state.
    Oracle { target_state: usize },
    /// Grover diffusion — reflect about the mean amplitude.
    Diffusion,
}

/// A quantum circuit: an ordered list of operations on `n` qubits.
#[derive(Debug, Clone)]
pub struct Circuit {
    pub num_qubits: usize,
    pub ops: Vec<Op>,
}

impl Circuit {
    /// Create an empty circuit with `n` qubits.
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            ops: Vec::new(),
        }
    }

    /// Append a single-qubit gate.
    pub fn add_gate1(&mut self, gate: Gate1, target: usize) -> &mut Self {
        assert!(target < self.num_qubits, "qubit index out of range");
        self.ops.push(Op::Gate1 { gate, target });
        self
    }

    /// Append a two-qubit gate.
    pub fn add_gate2(&mut self, gate: Gate2, control: usize, target: usize) -> &mut Self {
        assert!(control < self.num_qubits, "control qubit out of range");
        assert!(target < self.num_qubits, "target qubit out of range");
        assert_ne!(control, target, "control and target must differ");
        self.ops.push(Op::Gate2 {
            gate,
            control,
            target,
        });
        self
    }

    /// Append a measurement.
    pub fn add_measure(&mut self, target: usize) -> &mut Self {
        assert!(target < self.num_qubits, "qubit index out of range");
        self.ops.push(Op::Measure { target });
        self
    }

    // ── Convenience builders ────────────────────────────────────────

    pub fn h(&mut self, target: usize) -> &mut Self {
        self.add_gate1(gates::h(), target)
    }
    pub fn x(&mut self, target: usize) -> &mut Self {
        self.add_gate1(gates::x(), target)
    }
    pub fn y(&mut self, target: usize) -> &mut Self {
        self.add_gate1(gates::y(), target)
    }
    pub fn z(&mut self, target: usize) -> &mut Self {
        self.add_gate1(gates::z(), target)
    }
    pub fn s(&mut self, target: usize) -> &mut Self {
        self.add_gate1(gates::s(), target)
    }
    pub fn t(&mut self, target: usize) -> &mut Self {
        self.add_gate1(gates::t(), target)
    }
    pub fn cnot(&mut self, control: usize, target: usize) -> &mut Self {
        self.add_gate2(gates::cnot(), control, target)
    }
    pub fn cz(&mut self, control: usize, target: usize) -> &mut Self {
        self.add_gate2(gates::cz(), control, target)
    }
    pub fn swap(&mut self, a: usize, b: usize) -> &mut Self {
        self.add_gate2(gates::swap(), a, b)
    }
    pub fn oracle(&mut self, target_state: usize) -> &mut Self {
        self.ops.push(Op::Oracle { target_state });
        self
    }
    pub fn diffusion(&mut self) -> &mut Self {
        self.ops.push(Op::Diffusion);
        self
    }

    /// Execute the full circuit and return the final state plus measurement outcomes.
    pub fn run(&self) -> ExecutionResult {
        let mut state = StateVec::new(self.num_qubits);
        let mut measurements = Vec::new();
        // We track qubit handles internally to keep the move-only API honest.
        let mut qubits: Vec<Option<Qubit>> = state.qubits().into_iter().map(Some).collect();

        for op in &self.ops {
            match *op {
                Op::Gate1 { ref gate, target } => {
                    let q = qubits[target]
                        .take()
                        .expect("qubit already consumed or measured");
                    let q_new = state.apply1(gate, q);
                    qubits[target] = Some(q_new);
                }
                Op::Gate2 {
                    ref gate,
                    control,
                    target,
                } => {
                    let qc = qubits[control]
                        .take()
                        .expect("control qubit already consumed");
                    let qt = qubits[target]
                        .take()
                        .expect("target qubit already consumed");
                    let QubitPair(qc_new, qt_new) = state.apply2(gate, qc, qt);
                    qubits[control] = Some(qc_new);
                    qubits[target] = Some(qt_new);
                }
                Op::Measure { target } => {
                    let q = qubits[target]
                        .take()
                        .expect("qubit already consumed or measured");
                    let Measurement { outcome, qubit } = state.measure(q);
                    measurements.push((target, outcome));
                    qubits[target] = Some(qubit);
                }
                Op::Oracle { target_state } => {
                    crate::algorithms::oracle(&mut state, target_state);
                }
                Op::Diffusion => {
                    crate::algorithms::diffusion(&mut state);
                }
            }
        }

        ExecutionResult {
            state,
            measurements,
        }
    }
}

/// Result of running a circuit.
pub struct ExecutionResult {
    pub state: StateVec,
    /// (qubit_index, classical_outcome) for each measurement in order.
    pub measurements: Vec<(usize, bool)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bell_circuit() {
        let mut c = Circuit::new(2);
        c.h(0).cnot(0, 1);
        let result = c.run();
        let probs = result.state.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[3] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn ghz_three_qubits() {
        let mut c = Circuit::new(3);
        c.h(0).cnot(0, 1).cnot(0, 2);
        let result = c.run();
        let probs = result.state.probabilities();
        // GHZ: |000⟩ + |111⟩
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[7] - 0.5).abs() < 1e-10);
    }
}
