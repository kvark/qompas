//! State visualization panel — probability bars with phase-colored amplitudes.

use crate::state::StateVec;
use egui::{Color32, Pos2, Rect, Stroke, Vec2};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Bar chart height in logical pixels.
const BAR_MAX_HEIGHT: f32 = 120.0;
/// Width per basis state bar.
const BAR_WIDTH: f32 = 28.0;
/// Gap between bars.
const BAR_GAP: f32 = 4.0;

/// Map a complex phase angle to an HSV-style color (hue = phase, saturation = 1).
fn phase_color(amp: Complex64) -> Color32 {
    let phase = amp.arg(); // in [-PI, PI]
    let hue = ((phase + PI) / (2.0 * PI)) as f32; // normalized to [0, 1]
    let (r, g, b) = hsv_to_rgb(hue, 0.85, 0.95);
    Color32::from_rgb(r, g, b)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let h6 = h * 6.0;
    let i = h6.floor() as i32;
    let f = h6 - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    let (r, g, b) = match i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Draw a state-vector bar chart.
///
/// Each basis state gets a vertical bar whose height = probability and
/// whose color encodes the complex phase.
pub fn state_panel_ui(ui: &mut egui::Ui, state: &StateVec) {
    let n = state.num_qubits();
    let dim = 1 << n;
    let amps = state.amplitudes();

    ui.label(
        egui::RichText::new("State amplitudes")
            .strong()
            .size(14.0),
    );
    ui.add_space(4.0);

    // Phase legend
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        for deg in (0..360).step_by(60) {
            let hue = deg as f32 / 360.0;
            let (r, g, b) = hsv_to_rgb(hue, 0.85, 0.95);
            let color = Color32::from_rgb(r, g, b);
            ui.colored_label(color, format!("{deg}°"));
        }
        ui.label("(phase)");
    });
    ui.add_space(4.0);

    let total_width = dim as f32 * (BAR_WIDTH + BAR_GAP);
    let total_height = BAR_MAX_HEIGHT + 24.0; // +24 for labels
    let (_, painter_rect) = ui.allocate_space(Vec2::new(
        total_width.max(ui.available_width()),
        total_height,
    ));
    let painter = ui.painter();
    let origin = painter_rect.min;

    let baseline_y = origin.y + BAR_MAX_HEIGHT;

    for i in 0..dim {
        let amp = amps[i];
        let prob = amp.norm_sqr();
        let bar_h = prob as f32 * BAR_MAX_HEIGHT;
        let x = origin.x + i as f32 * (BAR_WIDTH + BAR_GAP);

        // Bar
        let bar_rect = Rect::from_min_size(
            Pos2::new(x, baseline_y - bar_h),
            Vec2::new(BAR_WIDTH, bar_h.max(1.0)),
        );

        let color = if prob > 1e-10 {
            phase_color(amp)
        } else {
            Color32::from_gray(50)
        };
        painter.rect_filled(bar_rect, 2.0, color);
        painter.rect_stroke(bar_rect, 2.0, Stroke::new(0.5, Color32::from_gray(100)));

        // Probability text
        if prob > 1e-4 {
            painter.text(
                Pos2::new(x + BAR_WIDTH / 2.0, baseline_y - bar_h - 2.0),
                egui::Align2::CENTER_BOTTOM,
                format!("{:.1}%", prob * 100.0),
                egui::FontId::proportional(10.0),
                Color32::from_gray(220),
            );
        }

        // Basis label
        painter.text(
            Pos2::new(x + BAR_WIDTH / 2.0, baseline_y + 4.0),
            egui::Align2::CENTER_TOP,
            format!("|{:0>width$b}⟩", i, width = n),
            egui::FontId::monospace(10.0),
            Color32::from_gray(180),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_color_is_deterministic() {
        let c1 = phase_color(Complex64::new(1.0, 0.0));
        let c2 = phase_color(Complex64::new(1.0, 0.0));
        assert_eq!(c1, c2);
    }

    #[test]
    fn phase_color_opposite_phases_differ() {
        let c_pos = phase_color(Complex64::new(1.0, 0.0));
        let c_neg = phase_color(Complex64::new(-1.0, 0.0));
        assert_ne!(c_pos, c_neg);
    }
}
