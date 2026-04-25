# qompas

**Quantum exploration library** — a state-vector simulator and interactive
visualizer for quantum circuits, written in Rust.

[![CI](https://github.com/kvark/qompas/actions/workflows/ci.yml/badge.svg)](https://github.com/kvark/qompas/actions/workflows/ci.yml)

Qompas pairs a fast simulator with a rich UI so you can build a circuit, watch
the state evolve gate-by-gate, and reason about superposition and entanglement
visually. It is built on [blade-graphics] for portable GPU compute and
[egui] for immediate-mode interaction.

[blade-graphics]: https://github.com/kvark/blade
[egui]: https://github.com/emilk/egui

## Why another simulator?

- **Borrow-checker–enforced quantum semantics.** A `Qubit` is a move-only
  handle: any attempt to duplicate one is rejected at compile time, mirroring
  the no-cloning theorem. Gates *consume* qubits and return new handles, so
  illegal circuit constructions do not type-check.
- **Portable GPU acceleration.** Large circuits (≥12 qubits) automatically
  dispatch onto the GPU via blade-graphics, which targets Vulkan / Metal /
  DX12 transparently. Small circuits stay on the CPU where dispatch overhead
  would dominate.
- **Built for exploration.** The egui-based UI lets you drag gates onto wires,
  step through execution, and watch a phase-colored bar chart of the
  amplitudes update in real time.
- **Standard formats.** A minimal OpenQASM 2.0 parser is included so you can
  load circuits exported from other tools.

## Features

- Single- and two-qubit gates: H, X, Y, Z, S, T, CNOT, CZ, SWAP
- Mid-circuit measurement
- Grover's search (oracle + diffusion) as a first-class circuit op
- CPU state-vector simulator (f64 amplitudes)
- GPU compute simulator (f32 amplitudes, WGSL kernels) with auto-selection
- OpenQASM 2.0 parser
- Interactive editor with drag-and-drop gate placement and right-click removal
- Phase-colored amplitude bar chart with HSV legend
- Step / run / pause / reset playback controls
- Headless rendering pipeline that can record demos as MP4 video

## Quick start

```bash
# Run the interactive viewer (defaults to a Bell circuit)
cargo run --release

# Run all tests
cargo test

# Render the Grover demo to grover.mp4 (requires ffmpeg)
cargo run --release --example grover_video
```

The binary opens a window with three panels:

- **Top:** preset menu (Bell, GHZ-3, Teleportation, Grover-4, empty)
- **Left:** circuit editor — drag a gate from the palette onto a qubit wire,
  right-click a placed gate to remove it
- **Center:** amplitude bar chart, colored by phase
- **Bottom:** step / run / pause / reset controls and speed slider

## Library usage

```rust
use qompas::circuit::Circuit;

// Build a Bell pair: |00⟩ → (|00⟩ + |11⟩)/√2
let mut c = Circuit::new(2);
c.h(0).cnot(0, 1);

let result = c.run();
let probs = result.state.probabilities();
assert!((probs[0] - 0.5).abs() < 1e-10);
assert!((probs[3] - 0.5).abs() < 1e-10);
```

Grover's search:

```rust
use qompas::{state::StateVec, algorithms};

let n = 4;
let target = 11; // search for |1011⟩
let iters = algorithms::optimal_iterations(n);

let mut state = StateVec::new(n);
algorithms::prepare(&mut state);
for _ in 0..iters {
    algorithms::iterate(&mut state, target);
}
assert!(state.probabilities()[target] > 0.9);
```

Loading OpenQASM:

```rust
use qompas::formats::openqasm;

let qasm = r#"
OPENQASM 2.0;
qreg q[2];
h q[0];
cx q[0], q[1];
"#;
let circuit = openqasm::parse(qasm).unwrap();
let result = circuit.run();
```

## GPU backend

`qompas::backend::Backend` picks between CPU and GPU automatically. The GPU
path becomes active above `GPU_THRESHOLD` qubits (default 12) when a
blade-graphics context is available. State amplitudes live in a device
storage buffer and gates are applied via WGSL compute shaders
(`src/shaders/quantum.wgsl`):

- `apply_gate1` — single-qubit gate, one thread per amplitude pair
- `apply_gate2` — two-qubit gate, one thread per amplitude quad
- `measure_prob` — workgroup reduction for measurement probabilities

The GPU backend works headlessly under [lavapipe] (Mesa's CPU Vulkan), which
is what the video recorder uses on machines without a discrete GPU.

[lavapipe]: https://docs.mesa3d.org/drivers/llvmpipe.html

## Demo: Grover's search, recorded

`examples/grover_video.rs` runs Grover's algorithm for `|10011⟩` on 5 qubits
and pipes RGB frames into `ffmpeg` to produce an MP4. The output shows the
amplitude of the marked state climbing from 1/√32 toward 1 over four
iterations, then over-rotating past the optimum on a fifth — a useful
intuition pump for why the iteration count matters.

```bash
cargo run --release --example grover_video
# → grover.mp4
```

## Repository layout

```
src/
  algorithms.rs     Grover's search (oracle, diffusion, optimal iterations)
  backend.rs        CPU/GPU selection and unified circuit execution
  circuit.rs        Circuit builder and Op enum
  formats/          OpenQASM 2.0 parser
  gates.rs          Standard gate definitions (Pauli, Hadamard, S, T, CNOT, CZ, SWAP)
  qubit.rs          Move-only Qubit handle
  state.rs          CPU state-vector simulator
  state_gpu.rs      GPU state-vector simulator (blade-graphics)
  shaders/          WGSL compute kernels
  viz/              egui panels: circuit editor, state bars, controls
  main.rs           Windowed application (winit + blade-graphics + blade-egui)
examples/
  grover_video.rs   Headless Grover render → MP4
```

## Development

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

CI runs the same three checks on Linux, macOS, and Windows.

## Status

Early. The simulator is correct and the visualizer is usable, but the API is
not stable. Expect breaking changes. Bug reports and PRs welcome.

## License

MIT — see [LICENSE](LICENSE).
