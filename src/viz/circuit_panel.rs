//! Interactive circuit diagram panel.
//!
//! Renders qubit wires and gate boxes in an egui panel. Gates can be added
//! from a palette via drag-and-drop, and removed by right-clicking.

use crate::circuit::Circuit;
use crate::viz::{layout_circuit, GateVisual};
use egui::{Color32, Pos2, Rect, Stroke, Vec2};

/// Size of a gate box in logical pixels.
const GATE_SIZE: f32 = 36.0;
/// Spacing between columns.
const COL_SPACING: f32 = 52.0;
/// Spacing between qubit wires.
const ROW_SPACING: f32 = 52.0;
/// Left margin for wire labels.
const LEFT_MARGIN: f32 = 48.0;
/// Top margin.
const TOP_MARGIN: f32 = 24.0;

/// The gate palette — what the user can drag onto the circuit.
const GATE_PALETTE: &[&str] = &["H", "X", "Y", "Z", "S", "T", "CNOT", "CZ", "SWAP", "M"];

/// Transient drag state for adding gates.
#[derive(Clone, Debug)]
pub struct DragState {
    /// Name of the gate being dragged from the palette.
    pub gate_name: String,
    /// Current pointer position (screen coords).
    pub pos: Pos2,
}

/// Persistent state for the circuit panel.
pub struct CircuitPanel {
    /// The circuit being edited.
    pub circuit: Circuit,
    /// Cached layout (recomputed on mutation).
    visuals: Vec<GateVisual>,
    /// Active drag from the palette.
    drag: Option<DragState>,
}

impl CircuitPanel {
    pub fn new(circuit: Circuit) -> Self {
        let visuals = layout_circuit(&circuit);
        Self {
            circuit,
            visuals,
            drag: None,
        }
    }

    /// Replace the circuit and refresh layout.
    pub fn set_circuit(&mut self, circuit: Circuit) {
        self.visuals = layout_circuit(&circuit);
        self.circuit = circuit;
    }

    fn refresh_layout(&mut self) {
        self.visuals = layout_circuit(&self.circuit);
    }

    /// Convert (column, row) to screen position.
    fn cell_pos(origin: Pos2, col: usize, row: usize) -> Pos2 {
        Pos2::new(
            origin.x + LEFT_MARGIN + col as f32 * COL_SPACING,
            origin.y + TOP_MARGIN + row as f32 * ROW_SPACING,
        )
    }

    /// Which qubit wire is closest to `y`?
    fn row_at_y(&self, origin: Pos2, y: f32) -> Option<usize> {
        let rel = y - origin.y - TOP_MARGIN + ROW_SPACING * 0.5;
        if rel < 0.0 {
            return None;
        }
        let row = (rel / ROW_SPACING) as usize;
        if row < self.circuit.num_qubits {
            Some(row)
        } else {
            None
        }
    }

    /// Show the circuit panel. Returns `true` if the circuit was mutated.
    pub fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut mutated = false;

        // ── Gate palette (horizontal toolbar at top) ────────────
        ui.horizontal(|ui| {
            ui.label("Gates:");
            for &name in GATE_PALETTE {
                let btn = ui.button(name);
                if btn.drag_started() {
                    self.drag = Some(DragState {
                        gate_name: name.to_string(),
                        pos: btn.rect.center(),
                    });
                }
            }
            if ui.button("+ Qubit").clicked() {
                self.circuit.num_qubits += 1;
                self.refresh_layout();
                mutated = true;
            }
        });

        ui.add_space(8.0);

        // ── Track pointer for drag ──────────────────────────────
        if let Some(ref mut drag) = self.drag {
            if let Some(pointer) = ui.ctx().pointer_latest_pos() {
                drag.pos = pointer;
            }
        }
        let drag_released = ui.input(|i| i.pointer.any_released());

        // ── Draw wires ──────────────────────────────────────────
        let wire_origin = ui.cursor().min;
        let depth = self.visuals.iter().map(|v| v.column + 1).max().unwrap_or(1);
        let diagram_width = LEFT_MARGIN + (depth as f32 + 1.0) * COL_SPACING;
        let diagram_height = TOP_MARGIN + self.circuit.num_qubits as f32 * ROW_SPACING;
        let _ = ui.allocate_space(Vec2::new(
            diagram_width.max(ui.available_width()),
            diagram_height,
        ));
        let painter = ui.painter();

        let wire_color = Color32::from_gray(160);
        for row in 0..self.circuit.num_qubits {
            let y = wire_origin.y + TOP_MARGIN + row as f32 * ROW_SPACING;
            painter.text(
                Pos2::new(wire_origin.x + 4.0, y),
                egui::Align2::LEFT_CENTER,
                format!("q{row}"),
                egui::FontId::monospace(14.0),
                Color32::from_gray(200),
            );
            let x_start = wire_origin.x + LEFT_MARGIN - 8.0;
            let x_end = wire_origin.x + LEFT_MARGIN + depth as f32 * COL_SPACING + 8.0;
            painter.line_segment(
                [Pos2::new(x_start, y), Pos2::new(x_end, y)],
                Stroke::new(1.0, wire_color),
            );
        }

        // ── Draw gates & collect right-click removals ───────────
        let mut remove_idx: Option<usize> = None;
        for (idx, vis) in self.visuals.iter().enumerate() {
            let center = Self::cell_pos(wire_origin, vis.column, vis.rows[0]);
            let rect = Rect::from_center_size(center, Vec2::splat(GATE_SIZE));

            // Two-qubit gate: vertical connector + second box
            if vis.rows.len() == 2 {
                let p0 = Self::cell_pos(wire_origin, vis.column, vis.rows[0]);
                let p1 = Self::cell_pos(wire_origin, vis.column, vis.rows[1]);
                painter.line_segment([p0, p1], Stroke::new(2.0, Color32::from_rgb(100, 180, 255)));
                let rect2 = Rect::from_center_size(p1, Vec2::splat(GATE_SIZE));
                painter.rect_filled(rect2, 4.0, Color32::from_rgb(40, 60, 90));
                painter.rect_stroke(
                    rect2,
                    4.0,
                    Stroke::new(1.0, Color32::from_rgb(100, 180, 255)),
                );
                painter.text(
                    p1,
                    egui::Align2::CENTER_CENTER,
                    &vis.label,
                    egui::FontId::monospace(12.0),
                    Color32::from_rgb(180, 220, 255),
                );
            }

            // Gate box
            let bg = if vis.label == "M" {
                Color32::from_rgb(90, 60, 40)
            } else if vis.label.starts_with('O') || vis.label == "Diff" {
                Color32::from_rgb(60, 40, 80)
            } else {
                Color32::from_rgb(40, 50, 80)
            };
            painter.rect_filled(rect, 4.0, bg);
            painter.rect_stroke(
                rect,
                4.0,
                Stroke::new(1.0, Color32::from_rgb(100, 160, 255)),
            );
            painter.text(
                center,
                egui::Align2::CENTER_CENTER,
                &vis.label,
                egui::FontId::monospace(14.0),
                Color32::WHITE,
            );

            // Right-click to remove (deferred)
            let resp = ui.interact(rect, egui::Id::new(("gate", idx)), egui::Sense::click());
            if resp.secondary_clicked() {
                remove_idx = Some(idx);
            }
        }

        // Apply deferred removal
        if let Some(idx) = remove_idx {
            if idx < self.circuit.ops.len() {
                self.circuit.ops.remove(idx);
                self.refresh_layout();
                mutated = true;
            }
        }

        // ── Handle drop from palette ────────────────────────────
        if drag_released {
            if let Some(drag) = self.drag.take() {
                if let Some(row) = self.row_at_y(wire_origin, drag.pos.y) {
                    mutated = self.place_gate(&drag.gate_name, row);
                }
            }
        }

        // ── Draw drag ghost ─────────────────────────────────────
        if let Some(ref drag) = self.drag {
            let ghost_rect = Rect::from_center_size(drag.pos, Vec2::splat(GATE_SIZE));
            painter.rect_filled(
                ghost_rect,
                4.0,
                Color32::from_rgba_unmultiplied(60, 80, 140, 180),
            );
            painter.text(
                drag.pos,
                egui::Align2::CENTER_CENTER,
                &drag.gate_name,
                egui::FontId::monospace(14.0),
                Color32::from_rgba_unmultiplied(255, 255, 255, 200),
            );
        }

        mutated
    }

    /// Place a gate from the palette onto the circuit at the given qubit row.
    fn place_gate(&mut self, name: &str, row: usize) -> bool {
        match name {
            "H" => {
                self.circuit.h(row);
            }
            "X" => {
                self.circuit.x(row);
            }
            "Y" => {
                self.circuit.y(row);
            }
            "Z" => {
                self.circuit.z(row);
            }
            "S" => {
                self.circuit.s(row);
            }
            "T" => {
                self.circuit.t(row);
            }
            "CNOT" | "CZ" | "SWAP" => {
                let target = if row + 1 < self.circuit.num_qubits {
                    row + 1
                } else {
                    0
                };
                if target == row {
                    return false;
                }
                match name {
                    "CNOT" => {
                        self.circuit.cnot(row, target);
                    }
                    "CZ" => {
                        self.circuit.cz(row, target);
                    }
                    "SWAP" => {
                        self.circuit.swap(row, target);
                    }
                    _ => unreachable!(),
                }
            }
            "M" => {
                self.circuit.add_measure(row);
            }
            _ => return false,
        }
        self.refresh_layout();
        true
    }
}
