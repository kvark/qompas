//! Top-level application state tying circuit editing, execution, and visualization together.

use crate::circuit::Circuit;
use crate::viz::circuit_panel::CircuitPanel;
use crate::viz::controls::{ExecController, PlayMode};
use crate::viz::state_panel;

/// Preset circuits available from the menu.
pub enum Preset {
    Bell,
    Ghz3,
    Teleportation,
    Grover4,
    Empty(usize),
}

impl Preset {
    pub fn build(&self) -> Circuit {
        match *self {
            Preset::Bell => {
                let mut c = Circuit::new(2);
                c.h(0).cnot(0, 1);
                c
            }
            Preset::Ghz3 => {
                let mut c = Circuit::new(3);
                c.h(0).cnot(0, 1).cnot(0, 2);
                c
            }
            Preset::Teleportation => {
                // Simplified teleportation circuit (3 qubits)
                let mut c = Circuit::new(3);
                // Create Bell pair between q1 and q2
                c.h(1).cnot(1, 2);
                // Alice's operations on q0 and q1
                c.cnot(0, 1).h(0);
                // Measurements
                c.add_measure(0);
                c.add_measure(1);
                c
            }
            Preset::Grover4 => {
                // 4-qubit Grover's search for |1011⟩ (index 11)
                let n = 4;
                let target_state = 11;
                let iters = crate::algorithms::optimal_iterations(n);
                let mut c = Circuit::new(n);
                // Hadamard on all qubits → uniform superposition
                for q in 0..n {
                    c.h(q);
                }
                // Grover iterations
                for _ in 0..iters {
                    c.oracle(target_state);
                    c.diffusion();
                }
                c
            }
            Preset::Empty(n) => Circuit::new(n),
        }
    }
}

/// The main application.
pub struct QompasApp {
    pub circuit_panel: CircuitPanel,
    pub exec: ExecController,
    /// Whether the exec state is stale (circuit was edited).
    needs_re_exec: bool,
}

impl QompasApp {
    /// Create the app with a default Bell-state circuit.
    pub fn new() -> Self {
        let circuit = Preset::Bell.build();
        let exec = ExecController::new(circuit.num_qubits);
        Self {
            circuit_panel: CircuitPanel::new(circuit),
            exec,
            needs_re_exec: true,
        }
    }

    /// Load a preset circuit.
    pub fn load_preset(&mut self, preset: Preset) {
        let circuit = preset.build();
        self.exec.reset(circuit.num_qubits);
        self.circuit_panel.set_circuit(circuit);
        self.needs_re_exec = true;
    }

    /// Main UI — call this from the egui frame callback.
    pub fn ui(&mut self, ctx: &egui::Context) {
        let dt = ctx.input(|i| i.stable_dt);

        // ── Top menu bar ────────────────────────────────────────
        egui::TopBottomPanel::top("menu").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("New (2 qubits)").clicked() {
                        self.load_preset(Preset::Empty(2));
                        ui.close_menu();
                    }
                    if ui.button("New (3 qubits)").clicked() {
                        self.load_preset(Preset::Empty(3));
                        ui.close_menu();
                    }
                });
                ui.menu_button("Presets", |ui| {
                    if ui.button("Bell state").clicked() {
                        self.load_preset(Preset::Bell);
                        ui.close_menu();
                    }
                    if ui.button("GHZ (3 qubits)").clicked() {
                        self.load_preset(Preset::Ghz3);
                        ui.close_menu();
                    }
                    if ui.button("Teleportation").clicked() {
                        self.load_preset(Preset::Teleportation);
                        ui.close_menu();
                    }
                    if ui.button("Grover (4 qubits)").clicked() {
                        self.load_preset(Preset::Grover4);
                        ui.close_menu();
                    }
                });
            });
        });

        // ── Bottom panel: execution controls ────────────────────
        egui::TopBottomPanel::bottom("controls").show(ctx, |ui| {
            let ctrl_changed = self.exec.ui(ui, &self.circuit_panel.circuit);
            if ctrl_changed {
                self.needs_re_exec = false;
            }
        });

        // ── Left panel: circuit editor ──────────────────────────
        egui::SidePanel::left("circuit")
            .default_width(400.0)
            .show(ctx, |ui| {
                ui.heading("Circuit");
                egui::ScrollArea::both().show(ui, |ui| {
                    let edited = self.circuit_panel.ui(ui);
                    if edited {
                        self.exec.reset(self.circuit_panel.circuit.num_qubits);
                        self.needs_re_exec = true;
                    }
                });
            });

        // ── Central panel: state visualization ──────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Quantum State");
            ui.add_space(8.0);

            // Auto-step if running
            self.exec.tick(&self.circuit_panel.circuit, dt);

            // If running, request continuous repaint
            if self.exec.mode == PlayMode::Running {
                ctx.request_repaint();
            }

            state_panel::state_panel_ui(ui, &self.exec.state);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_build_valid_circuits() {
        for preset in [
            Preset::Bell,
            Preset::Ghz3,
            Preset::Teleportation,
            Preset::Empty(4),
        ] {
            let c = preset.build();
            let result = c.run();
            let probs = result.state.probabilities();
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "probabilities don't sum to 1");
        }
    }

    #[test]
    fn app_load_preset_resets_state() {
        let mut app = QompasApp::new();
        app.exec.step_forward(&app.circuit_panel.circuit);
        assert_eq!(app.exec.step, 1);
        app.load_preset(Preset::Ghz3);
        assert_eq!(app.exec.step, 0);
        assert_eq!(app.circuit_panel.circuit.num_qubits, 3);
    }
}
