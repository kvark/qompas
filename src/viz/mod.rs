//! Visualization scaffolding for quantum circuit exploration.
//!
//! This module will provide:
//! - Circuit diagram rendering via egui
//! - State-vector bar charts (amplitude + phase)
//! - Bloch sphere display for single-qubit states
//! - Interactive step-through controls
//!
//! The rendering backend is **blade-graphics**, giving us portable GPU
//! acceleration across Vulkan, Metal, and DX12.

use crate::circuit::{Circuit, Op};

/// Visual layout information for one gate in the circuit diagram.
#[derive(Debug, Clone)]
pub struct GateVisual {
    /// Column (time step) in the circuit diagram.
    pub column: usize,
    /// Row(s) — qubit wire indices this gate touches.
    pub rows: Vec<usize>,
    /// Display label.
    pub label: String,
}

/// Lay out a circuit into a column-based visual representation.
pub fn layout_circuit(circuit: &Circuit) -> Vec<GateVisual> {
    // Simple greedy layout: assign each gate to the earliest column
    // where all its qubit wires are free.
    let mut wire_horizon = vec![0usize; circuit.num_qubits];
    let mut visuals = Vec::with_capacity(circuit.ops.len());

    for op in &circuit.ops {
        let (label, rows) = match op {
            Op::Gate1 { gate, target } => (gate.name.to_string(), vec![*target]),
            Op::Gate2 {
                gate,
                control,
                target,
            } => (gate.name.to_string(), vec![*control, *target]),
            Op::Measure { target } => ("M".to_string(), vec![*target]),
        };

        let col = rows.iter().map(|&r| wire_horizon[r]).max().unwrap_or(0);
        for &r in &rows {
            wire_horizon[r] = col + 1;
        }

        visuals.push(GateVisual {
            column: col,
            rows,
            label,
        });
    }

    visuals
}

/// Total number of columns (time steps) in the layout.
pub fn circuit_depth(visuals: &[GateVisual]) -> usize {
    visuals.iter().map(|v| v.column + 1).max().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bell_layout() {
        let mut c = Circuit::new(2);
        c.h(0).cnot(0, 1);
        let vis = layout_circuit(&c);
        assert_eq!(vis.len(), 2);
        assert_eq!(vis[0].column, 0); // H on q0
        assert_eq!(vis[1].column, 1); // CNOT after H
        assert_eq!(circuit_depth(&vis), 2);
    }

    #[test]
    fn parallel_gates() {
        let mut c = Circuit::new(3);
        c.h(0).h(1).h(2);
        let vis = layout_circuit(&c);
        // All three H gates operate on different qubits → same column
        assert!(vis.iter().all(|v| v.column == 0));
        assert_eq!(circuit_depth(&vis), 1);
    }
}
