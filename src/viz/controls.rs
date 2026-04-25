//! Execution controls — step, run, reset, speed slider.
//!
//! Manages a "playhead" through the circuit's operation list so users
//! can step through execution one gate at a time, or run the whole thing.

use crate::circuit::{Circuit, Op};
use crate::qubit::{Measurement, Qubit, QubitPair};
use crate::state::StateVec;

/// Execution mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayMode {
    /// Paused — waiting for user to step or press run.
    Paused,
    /// Auto-advancing at the configured speed.
    Running,
}

/// Execution controller for a quantum circuit.
pub struct ExecController {
    /// Current step index (0 = before first gate, ops.len() = done).
    pub step: usize,
    /// Simulation state at current step.
    pub state: StateVec,
    /// Qubit handles (tracked for move semantics).
    qubits: Vec<Option<Qubit>>,
    /// Collected measurement results.
    pub measurements: Vec<(usize, bool)>,
    /// Playback mode.
    pub mode: PlayMode,
    /// Delay between auto-steps in seconds.
    pub step_delay: f32,
    /// Accumulator for timing.
    pub time_acc: f32,
}

impl ExecController {
    /// Create a new controller for a circuit with `n` qubits.
    pub fn new(num_qubits: usize) -> Self {
        let state = StateVec::new(num_qubits);
        let qubits = state.qubits().into_iter().map(Some).collect();
        Self {
            step: 0,
            state,
            qubits,
            measurements: Vec::new(),
            mode: PlayMode::Paused,
            step_delay: 0.5,
            time_acc: 0.0,
        }
    }

    /// Reset to the initial |0…0⟩ state.
    pub fn reset(&mut self, num_qubits: usize) {
        self.state = StateVec::new(num_qubits);
        self.qubits = self.state.qubits().into_iter().map(Some).collect();
        self.step = 0;
        self.measurements.clear();
        self.mode = PlayMode::Paused;
        self.time_acc = 0.0;
    }

    /// Is execution complete?
    pub fn is_done(&self, circuit: &Circuit) -> bool {
        self.step >= circuit.ops.len()
    }

    /// Advance one step. Returns `true` if a step was taken.
    pub fn step_forward(&mut self, circuit: &Circuit) -> bool {
        if self.step >= circuit.ops.len() {
            self.mode = PlayMode::Paused;
            return false;
        }

        let op = &circuit.ops[self.step];
        match *op {
            Op::Gate1 { ref gate, target } => {
                let q = self.qubits[target].take().expect("qubit consumed");
                let q_new = self.state.apply1(gate, q);
                self.qubits[target] = Some(q_new);
            }
            Op::Gate2 {
                ref gate,
                control,
                target,
            } => {
                let qc = self.qubits[control].take().expect("control consumed");
                let qt = self.qubits[target].take().expect("target consumed");
                let QubitPair(qc_new, qt_new) = self.state.apply2(gate, qc, qt);
                self.qubits[control] = Some(qc_new);
                self.qubits[target] = Some(qt_new);
            }
            Op::Measure { target } => {
                let q = self.qubits[target].take().expect("qubit consumed");
                let Measurement { outcome, qubit } = self.state.measure(q);
                self.measurements.push((target, outcome));
                self.qubits[target] = Some(qubit);
            }
            Op::Oracle { target_state } => {
                crate::algorithms::oracle(&mut self.state, target_state);
            }
            Op::Diffusion => {
                crate::algorithms::diffusion(&mut self.state);
            }
        }

        self.step += 1;
        true
    }

    /// Tick the auto-play timer. Call every frame with `dt` in seconds.
    pub fn tick(&mut self, circuit: &Circuit, dt: f32) {
        if self.mode != PlayMode::Running {
            return;
        }
        self.time_acc += dt;
        while self.time_acc >= self.step_delay {
            self.time_acc -= self.step_delay;
            if !self.step_forward(circuit) {
                break;
            }
        }
    }

    /// Draw the controls UI. Returns `true` if state changed.
    pub fn ui(&mut self, ui: &mut egui::Ui, circuit: &Circuit) -> bool {
        let mut changed = false;

        ui.horizontal(|ui| {
            if ui.button("⏮ Reset").clicked() {
                self.reset(circuit.num_qubits);
                changed = true;
            }

            let step_btn = ui.add_enabled(!self.is_done(circuit), egui::Button::new("⏭ Step"));
            if step_btn.clicked() {
                self.mode = PlayMode::Paused;
                changed = self.step_forward(circuit);
            }

            let play_label = match self.mode {
                PlayMode::Paused => "▶ Run",
                PlayMode::Running => "⏸ Pause",
            };
            if ui.button(play_label).clicked() {
                self.mode = match self.mode {
                    PlayMode::Paused => PlayMode::Running,
                    PlayMode::Running => PlayMode::Paused,
                };
            }

            ui.separator();
            ui.label(format!("Step {}/{}", self.step, circuit.ops.len()));

            if !self.measurements.is_empty() {
                ui.separator();
                let bits: String = self
                    .measurements
                    .iter()
                    .map(|&(q, v)| format!("q{}={}", q, v as u8))
                    .collect::<Vec<_>>()
                    .join(" ");
                ui.label(format!("Measured: {bits}"));
            }
        });

        ui.horizontal(|ui| {
            ui.label("Speed:");
            let slider = egui::Slider::new(&mut self.step_delay, 0.05..=2.0)
                .suffix("s")
                .logarithmic(true);
            ui.add(slider);
        });

        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_through_bell() {
        let mut c = Circuit::new(2);
        c.h(0).cnot(0, 1);

        let mut ctrl = ExecController::new(2);
        assert!(!ctrl.is_done(&c));
        ctrl.step_forward(&c); // H
        assert_eq!(ctrl.step, 1);
        ctrl.step_forward(&c); // CNOT
        assert_eq!(ctrl.step, 2);
        assert!(ctrl.is_done(&c));

        let probs = ctrl.state.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[3] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn reset_clears_state() {
        let mut c = Circuit::new(1);
        c.x(0);

        let mut ctrl = ExecController::new(1);
        ctrl.step_forward(&c);
        assert!((ctrl.state.probabilities()[1] - 1.0).abs() < 1e-10);

        ctrl.reset(1);
        assert_eq!(ctrl.step, 0);
        assert!((ctrl.state.probabilities()[0] - 1.0).abs() < 1e-10);
    }
}
