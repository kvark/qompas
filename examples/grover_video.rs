//! Grover's search algorithm — animated video demo.
//!
//! Runs a 5-qubit Grover search, renders each iteration as a frame, and pipes
//! raw RGB data to ffmpeg to produce an MP4 video.
//!
//! ```sh
//! cargo run --example grover_video --release
//! # → outputs grover.mp4
//! ```

use num_complex::Complex64;
use qompas::algorithms;
use qompas::state::StateVec;
use std::f64::consts::PI;
use std::io::Write;
use std::process::{Command, Stdio};

// ── Video parameters ──────────────────────────────────────────────
const WIDTH: usize = 1280;
const HEIGHT: usize = 720;
const FPS: usize = 30;

/// Frames to interpolate between algorithm snapshots.
const TRANSITION_FRAMES: usize = 40;
/// Frames to hold at each snapshot.
const HOLD_FRAMES: usize = 25;

// ── Layout ────────────────────────────────────────────────────────
const BAR_AREA_LEFT: usize = 100;
const BAR_AREA_RIGHT: usize = 1180;
const BAR_AREA_TOP: usize = 180;
const BAR_AREA_BOTTOM: usize = 600;
const LABEL_Y: usize = 620;
const BAR_GAP: usize = 2;

// ── Colors ────────────────────────────────────────────────────────
const BG: [u8; 3] = [10, 14, 39];
const GRID_COLOR: [u8; 3] = [30, 34, 60];
const TARGET_GLOW: [u8; 3] = [255, 215, 0];

// ── Simple framebuffer ───────────────────────────────────────────

struct Canvas {
    pixels: Vec<u8>, // RGB, row-major
    width: usize,
    height: usize,
}

impl Canvas {
    fn new(width: usize, height: usize) -> Self {
        Self {
            pixels: vec![0; width * height * 3],
            width,
            height,
        }
    }

    fn clear(&mut self, color: [u8; 3]) {
        for chunk in self.pixels.chunks_exact_mut(3) {
            chunk.copy_from_slice(&color);
        }
    }

    fn set_pixel(&mut self, x: usize, y: usize, color: [u8; 3]) {
        if x < self.width && y < self.height {
            let idx = (y * self.width + x) * 3;
            self.pixels[idx] = color[0];
            self.pixels[idx + 1] = color[1];
            self.pixels[idx + 2] = color[2];
        }
    }

    fn fill_rect(&mut self, x: usize, y: usize, w: usize, h: usize, color: [u8; 3]) {
        for dy in 0..h {
            for dx in 0..w {
                self.set_pixel(x + dx, y + dy, color);
            }
        }
    }

    /// Draw a rect with 1px outline.
    fn stroke_rect(&mut self, x: usize, y: usize, w: usize, h: usize, color: [u8; 3]) {
        for dx in 0..w {
            self.set_pixel(x + dx, y, color);
            self.set_pixel(x + dx, y + h.saturating_sub(1), color);
        }
        for dy in 0..h {
            self.set_pixel(x, y + dy, color);
            self.set_pixel(x + w.saturating_sub(1), y + dy, color);
        }
    }

    fn hline(&mut self, x0: usize, x1: usize, y: usize, color: [u8; 3]) {
        for x in x0..x1 {
            self.set_pixel(x, y, color);
        }
    }
}

// ── Bitmap font (5x7) ────────────────────────────────────────────

// Minimal bitmap font for digits, basic letters, and symbols.
// Each glyph is 5 columns × 7 rows, stored as 7 bytes (each byte's low 5 bits = 1 row).
#[rustfmt::skip]
const FONT: &[(char, [u8; 7])] = &[
    ('0', [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110]),
    ('1', [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110]),
    ('2', [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111]),
    ('3', [0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110]),
    ('4', [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010]),
    ('5', [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110]),
    ('6', [0b01110, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b01110]),
    ('7', [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000]),
    ('8', [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110]),
    ('9', [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b10001, 0b01110]),
    ('.', [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100]),
    ('%', [0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011]),
    ('|', [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100]),
    (' ', [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000]),
    (':', [0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000]),
    ('/', [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000]),
    ('-', [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000]),
    ('G', [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110]),
    ('r', [0b00000, 0b00000, 0b10110, 0b11001, 0b10000, 0b10000, 0b10000]),
    ('o', [0b00000, 0b00000, 0b01110, 0b10001, 0b10001, 0b10001, 0b01110]),
    ('v', [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100]),
    ('e', [0b00000, 0b00000, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110]),
    ('s', [0b00000, 0b00000, 0b01111, 0b10000, 0b01110, 0b00001, 0b11110]),
    ('S', [0b01110, 0b10001, 0b10000, 0b01110, 0b00001, 0b10001, 0b01110]),
    ('a', [0b00000, 0b00000, 0b01110, 0b00001, 0b01111, 0b10001, 0b01111]),
    ('c', [0b00000, 0b00000, 0b01110, 0b10000, 0b10000, 0b10001, 0b01110]),
    ('h', [0b10000, 0b10000, 0b10110, 0b11001, 0b10001, 0b10001, 0b10001]),
    ('i', [0b00100, 0b00000, 0b01100, 0b00100, 0b00100, 0b00100, 0b01110]),
    ('n', [0b00000, 0b00000, 0b10110, 0b11001, 0b10001, 0b10001, 0b10001]),
    ('g', [0b00000, 0b00000, 0b01111, 0b10001, 0b01111, 0b00001, 0b01110]),
    ('f', [0b00110, 0b01001, 0b01000, 0b11100, 0b01000, 0b01000, 0b01000]),
    ('t', [0b01000, 0b01000, 0b11100, 0b01000, 0b01000, 0b01001, 0b00110]),
    ('I', [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110]),
    ('P', [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000]),
    ('b', [0b10000, 0b10000, 0b10110, 0b11001, 0b10001, 0b10001, 0b11110]),
    ('u', [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b10011, 0b01101]),
    ('q', [0b00000, 0b00000, 0b01101, 0b10011, 0b01111, 0b00001, 0b00001]),
    ('p', [0b00000, 0b00000, 0b11110, 0b10001, 0b11110, 0b10000, 0b10000]),
    ('d', [0b00001, 0b00001, 0b01101, 0b10011, 0b10001, 0b10001, 0b01111]),
    ('T', [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100]),
    ('l', [0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110]),
    ('m', [0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10001, 0b10001]),
    ('U', [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110]),
    ('N', [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001]),
    ('F', [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000]),
    ('O', [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110]),
    ('R', [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001]),
    ('w', [0b00000, 0b00000, 0b10001, 0b10001, 0b10101, 0b10101, 0b01010]),
    ('y', [0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01000, 0b10000]),
    ('k', [0b10000, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001]),
    ('A', [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001]),
    ('D', [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110]),
    ('x', [0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001]),
];

fn draw_char(canvas: &mut Canvas, ch: char, x: usize, y: usize, color: [u8; 3], scale: usize) {
    let glyph = FONT.iter().find(|(c, _)| *c == ch);
    let rows = match glyph {
        Some((_, rows)) => rows,
        None => return, // skip unknown chars
    };
    for (row_idx, &row_bits) in rows.iter().enumerate() {
        for col in 0..5 {
            if row_bits & (1 << (4 - col)) != 0 {
                for sy in 0..scale {
                    for sx in 0..scale {
                        canvas.set_pixel(x + col * scale + sx, y + row_idx * scale + sy, color);
                    }
                }
            }
        }
    }
}

fn draw_text(canvas: &mut Canvas, text: &str, x: usize, y: usize, color: [u8; 3], scale: usize) {
    let char_spacing = 6 * scale; // 5px glyph + 1px gap, scaled
    for (i, ch) in text.chars().enumerate() {
        draw_char(canvas, ch, x + i * char_spacing, y, color, scale);
    }
}

/// Centered text drawing.
fn draw_text_centered(canvas: &mut Canvas, text: &str, cx: usize, y: usize, color: [u8; 3], scale: usize) {
    let w = text.len() * 6 * scale;
    let x = cx.saturating_sub(w / 2);
    draw_text(canvas, text, x, y, color, scale);
}

// ── Phase → color (same as state_panel.rs) ───────────────────────

fn phase_color(amp: Complex64) -> [u8; 3] {
    let phase = amp.arg();
    let hue = ((phase + PI) / (2.0 * PI)) as f32;
    hsv_to_rgb(hue, 0.85, 0.95)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [u8; 3] {
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
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

/// Smoothstep easing.
fn smoothstep(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

/// Linearly interpolate two amplitude vectors.
fn lerp_amps(a: &[Complex64], b: &[Complex64], t: f64) -> Vec<Complex64> {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| a * (1.0 - t) + b * t)
        .collect()
}

// ── Frame rendering ──────────────────────────────────────────────

fn render_frame(
    canvas: &mut Canvas,
    amps: &[Complex64],
    num_qubits: usize,
    target_state: usize,
    title: &str,
    subtitle: &str,
) {
    canvas.clear(BG);

    let dim = 1 << num_qubits;
    let bar_area_width = BAR_AREA_RIGHT - BAR_AREA_LEFT;
    let bar_area_height = BAR_AREA_BOTTOM - BAR_AREA_TOP;
    let bar_w = (bar_area_width - (dim - 1) * BAR_GAP) / dim;

    // Grid lines
    for frac in [0.25, 0.50, 0.75, 1.00] {
        let y = BAR_AREA_BOTTOM - (frac * bar_area_height as f64) as usize;
        canvas.hline(BAR_AREA_LEFT, BAR_AREA_RIGHT, y, GRID_COLOR);
    }

    // Bars
    for i in 0..dim {
        let amp = amps[i];
        let prob = amp.norm_sqr();
        let bar_h = (prob * bar_area_height as f64) as usize;
        let x = BAR_AREA_LEFT + i * (bar_w + BAR_GAP);
        let y = BAR_AREA_BOTTOM - bar_h;

        let color = if prob > 1e-10 {
            phase_color(amp)
        } else {
            [50, 50, 50]
        };

        canvas.fill_rect(x, y, bar_w, bar_h, color);

        // Target state gets a golden outline
        if i == target_state && prob > 0.01 {
            canvas.stroke_rect(x.saturating_sub(1), y.saturating_sub(1), bar_w + 2, bar_h + 2, TARGET_GLOW);
        }

        // Probability label above bar (only if > 1%)
        if prob > 0.01 {
            let pct_text = format!("{:.1}%", prob * 100.0);
            let text_x = x + bar_w / 2;
            let text_y = y.saturating_sub(12);
            draw_text_centered(canvas, &pct_text, text_x, text_y, [200, 200, 220], 1);
        }

        // Basis label below
        let label = format!("|{:0>width$b}", i, width = num_qubits);
        let label_x = x + bar_w / 2;
        draw_text_centered(canvas, &label, label_x, LABEL_Y, [140, 140, 160], 1);
    }

    // Title and subtitle
    draw_text_centered(canvas, title, WIDTH / 2, 30, [220, 220, 240], 3);
    draw_text_centered(canvas, subtitle, WIDTH / 2, 80, [160, 160, 180], 2);

    // Target indicator
    let target_label = format!("Target: |{:0>width$b}", target_state, width = num_qubits);
    draw_text(canvas, &target_label, 40, HEIGHT - 50, TARGET_GLOW, 2);

    // Probability readout
    let prob = amps[target_state].norm_sqr();
    let prob_text = format!("P: {:.1}%", prob * 100.0);
    draw_text(canvas, &prob_text, 40, HEIGHT - 25, [200, 200, 220], 2);
}

fn main() {
    let num_qubits = 5;
    let target_state = 19; // |10011⟩
    let num_iterations = algorithms::optimal_iterations(num_qubits);

    println!("Grover's Search — {num_qubits} qubits, target |{target_state:0>5b}⟩");
    println!("Optimal iterations: {num_iterations}");
    println!("Rendering video...");

    // Collect snapshots: initial, uniform, each iteration, and one over-rotation
    let mut snapshots: Vec<(Vec<Complex64>, String, String)> = Vec::new();

    // Snapshot 0: |00000⟩
    let state0 = StateVec::new(num_qubits);
    snapshots.push((
        state0.amplitudes().to_vec(),
        "Grovers Search".to_string(),
        "Initial state".to_string(),
    ));

    // Snapshot 1: uniform superposition after H⊗n
    let mut state = StateVec::new(num_qubits);
    algorithms::prepare(&mut state);
    snapshots.push((
        state.amplitudes().to_vec(),
        "Grovers Search".to_string(),
        "Uniform superposition".to_string(),
    ));

    // Snapshots 2..n+1: Grover iterations
    for k in 1..=num_iterations {
        algorithms::iterate(&mut state, target_state);
        snapshots.push((
            state.amplitudes().to_vec(),
            "Grovers Search".to_string(),
            format!("Iteration {k}/{num_iterations}"),
        ));
    }

    // Snapshot n+2: over-rotation (one extra iteration)
    algorithms::iterate(&mut state, target_state);
    snapshots.push((
        state.amplitudes().to_vec(),
        "Grovers Search".to_string(),
        "Over-rotation".to_string(),
    ));

    // Total frames
    let num_transitions = snapshots.len() - 1;
    let total_frames =
        HOLD_FRAMES + num_transitions * (TRANSITION_FRAMES + HOLD_FRAMES) + HOLD_FRAMES;
    println!("Total frames: {total_frames} ({:.1}s at {FPS}fps)", total_frames as f64 / FPS as f64);

    // Launch ffmpeg
    let output_path = "grover.mp4";
    let mut ffmpeg = Command::new("ffmpeg")
        .args([
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", &format!("{WIDTH}x{HEIGHT}"),
            "-r", &FPS.to_string(),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to launch ffmpeg");

    let stdin = ffmpeg.stdin.as_mut().expect("failed to open ffmpeg stdin");
    let mut canvas = Canvas::new(WIDTH, HEIGHT);
    let mut frames_written = 0usize;

    // Helper: write N hold frames
    let write_hold = |stdin: &mut dyn Write, canvas: &Canvas, n: usize, written: &mut usize| {
        for _ in 0..n {
            stdin.write_all(&canvas.pixels).unwrap();
            *written += 1;
        }
    };

    // Initial hold
    render_frame(
        &mut canvas,
        &snapshots[0].0,
        num_qubits,
        target_state,
        &snapshots[0].1,
        &snapshots[0].2,
    );
    write_hold(stdin, &canvas, HOLD_FRAMES, &mut frames_written);

    // Transitions + holds
    for i in 0..num_transitions {
        let (amps_a, _, _) = &snapshots[i];
        let (amps_b, title_b, sub_b) = &snapshots[i + 1];

        // Transition frames with smooth interpolation
        for f in 0..TRANSITION_FRAMES {
            let t_raw = (f + 1) as f64 / TRANSITION_FRAMES as f64;
            let t = smoothstep(t_raw);
            let interp = lerp_amps(amps_a, amps_b, t);

            // Show destination labels during transition
            render_frame(&mut canvas, &interp, num_qubits, target_state, title_b, sub_b);
            stdin.write_all(&canvas.pixels).unwrap();
            frames_written += 1;
        }

        // Hold at destination
        render_frame(&mut canvas, amps_b, num_qubits, target_state, title_b, sub_b);
        write_hold(stdin, &canvas, HOLD_FRAMES, &mut frames_written);
    }

    // Extra hold at the end
    let last = snapshots.last().unwrap();
    render_frame(&mut canvas, &last.0, num_qubits, target_state, &last.1, &last.2);
    write_hold(stdin, &canvas, HOLD_FRAMES, &mut frames_written);

    drop(ffmpeg.stdin.take());
    let output = ffmpeg.wait_with_output().expect("ffmpeg failed");

    if output.status.success() {
        println!("Wrote {output_path} ({frames_written} frames, {:.1}s)", frames_written as f64 / FPS as f64);
    } else {
        eprintln!("ffmpeg error: {}", String::from_utf8_lossy(&output.stderr));
        std::process::exit(1);
    }
}
