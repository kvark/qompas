//! Backend abstraction — automatic CPU/GPU selection.
//!
//! [`Backend`] wraps both the CPU [`StateVec`] and GPU [`StateGpu`] simulators
//! behind a unified interface. It picks the GPU path when a blade-graphics
//! context is available and the qubit count is large enough to benefit from
//! compute shader dispatch.

use crate::circuit::{Circuit, ExecutionResult, Op};
use crate::state::StateVec;
use crate::state_gpu::{QuantumPipelines, StateGpu};

/// Minimum qubit count where GPU dispatch starts paying off.
/// Below this, the CPU path is faster due to dispatch overhead.
pub const GPU_THRESHOLD: usize = 12;

/// GPU resources, lazily initialised.
pub struct GpuResources {
    pub context: blade_graphics::Context,
    pub pipelines: QuantumPipelines,
}

impl GpuResources {
    /// Try to initialise GPU resources. Returns `None` if no GPU is available.
    pub fn try_init() -> Option<Self> {
        let desc = blade_graphics::ContextDesc {
            presentation: false, // headless compute only
            validation: cfg!(debug_assertions),
            ..Default::default()
        };
        let context = unsafe { blade_graphics::Context::init(desc) }.ok()?;
        let pipelines = QuantumPipelines::new(&context);
        log::info!(
            "GPU compute backend ready: {}",
            context.device_information().device_name
        );
        Some(Self { context, pipelines })
    }

    pub fn destroy(&mut self) {
        self.pipelines.destroy(&self.context);
    }
}

/// Which backend is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Gpu,
}

/// Unified simulation backend.
pub enum Backend {
    Cpu {
        state: StateVec,
    },
    Gpu {
        state_gpu: StateGpu,
        /// We also keep a CPU copy for readback / viz.
        state_cpu: StateVec,
        dirty: bool,
    },
}

impl Backend {
    /// Create a new backend for `n` qubits.
    /// Uses GPU if resources are provided and qubit count is above threshold.
    pub fn new(num_qubits: usize, gpu: Option<&GpuResources>) -> Self {
        if num_qubits >= GPU_THRESHOLD {
            if let Some(res) = gpu {
                let state_gpu = StateGpu::new(num_qubits, &res.context);
                return Backend::Gpu {
                    state_gpu,
                    state_cpu: StateVec::new(num_qubits),
                    dirty: true,
                };
            }
        }
        Backend::Cpu {
            state: StateVec::new(num_qubits),
        }
    }

    pub fn kind(&self) -> BackendKind {
        match *self {
            Backend::Cpu { .. } => BackendKind::Cpu,
            Backend::Gpu { .. } => BackendKind::Gpu,
        }
    }

    pub fn num_qubits(&self) -> usize {
        match *self {
            Backend::Cpu { ref state } => state.num_qubits(),
            Backend::Gpu { ref state_cpu, .. } => state_cpu.num_qubits(),
        }
    }

    /// Get the CPU-side state (for visualization). On GPU backend, call
    /// `sync_to_cpu` first to ensure it's up to date.
    pub fn cpu_state(&self) -> &StateVec {
        match *self {
            Backend::Cpu { ref state } => state,
            Backend::Gpu { ref state_cpu, .. } => state_cpu,
        }
    }

    /// Run a full circuit on the appropriate backend.
    pub fn run_circuit(
        num_qubits: usize,
        circuit: &Circuit,
        gpu: Option<&GpuResources>,
    ) -> ExecutionResult {
        if num_qubits >= GPU_THRESHOLD {
            if let Some(res) = gpu {
                return Self::run_circuit_gpu(num_qubits, circuit, res);
            }
        }
        // CPU path
        circuit.run()
    }

    /// Run a circuit entirely on the GPU.
    fn run_circuit_gpu(
        num_qubits: usize,
        circuit: &Circuit,
        res: &GpuResources,
    ) -> ExecutionResult {
        let gpu = &res.context;
        let pipelines = &res.pipelines;
        let state_gpu = StateGpu::new(num_qubits, gpu);

        // Upload |0…0⟩
        let mut encoder = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
            name: "init",
            buffer_count: 2,
        });
        encoder.start();
        state_gpu.upload_initial(&mut encoder);
        let sp = gpu.submit(&mut encoder);
        gpu.wait_for(&sp, !0);
        gpu.destroy_command_encoder(&mut encoder);

        let mut measurements = Vec::new();

        // Execute ops, batching consecutive gates into a single submit
        let mut encoder = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
            name: "circuit",
            buffer_count: 2,
        });
        encoder.start();

        for op in &circuit.ops {
            match *op {
                Op::Gate1 { ref gate, target } => {
                    state_gpu.dispatch_gate1(&mut encoder, pipelines, gate, target);
                }
                Op::Gate2 {
                    ref gate,
                    control,
                    target,
                } => {
                    state_gpu.dispatch_gate2(&mut encoder, pipelines, gate, control, target);
                }
                Op::Measure { target } => {
                    // Measurement needs a sync round-trip
                    state_gpu.dispatch_measure_prob(&mut encoder, pipelines, target);
                    let sp = gpu.submit(&mut encoder);
                    gpu.wait_for(&sp, !0);
                    gpu.destroy_command_encoder(&mut encoder);

                    let prob0 = state_gpu.read_prob0(gpu);
                    let outcome = prob0 < 0.5;
                    measurements.push((target, outcome));

                    // TODO: dispatch a collapse kernel here.
                    // For now, measurement is approximate (prob only).

                    encoder = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
                        name: "circuit_cont",
                        buffer_count: 2,
                    });
                    encoder.start();
                }
                Op::Oracle { .. } | Op::Diffusion => {
                    // These n-qubit ops need CPU fallback — download, apply, re-upload
                    state_gpu.download(&mut encoder);
                    let sp = gpu.submit(&mut encoder);
                    gpu.wait_for(&sp, !0);
                    gpu.destroy_command_encoder(&mut encoder);

                    let amps_f32 = state_gpu.read_amplitudes(gpu);
                    let mut cpu_state = StateVec::new(num_qubits);
                    for (i, &[re, im]) in amps_f32.iter().enumerate() {
                        cpu_state.amplitudes[i] = num_complex::Complex64::new(re as f64, im as f64);
                    }
                    match *op {
                        Op::Oracle { target_state } => {
                            crate::algorithms::oracle(&mut cpu_state, target_state);
                        }
                        Op::Diffusion => {
                            crate::algorithms::diffusion(&mut cpu_state);
                        }
                        _ => unreachable!(),
                    }
                    // Re-upload
                    unsafe {
                        let ptr = state_gpu.staging_buffer_ptr() as *mut [f32; 2];
                        for (i, amp) in cpu_state.amplitudes.iter().enumerate() {
                            *ptr.add(i) = [amp.re as f32, amp.im as f32];
                        }
                    }
                    encoder = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
                        name: "circuit_cont",
                        buffer_count: 2,
                    });
                    encoder.start();
                    state_gpu.upload_initial(&mut encoder);
                    let sp = gpu.submit(&mut encoder);
                    gpu.wait_for(&sp, !0);
                    gpu.destroy_command_encoder(&mut encoder);

                    encoder = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
                        name: "circuit_cont2",
                        buffer_count: 2,
                    });
                    encoder.start();
                }
            }
        }

        // Download final state
        state_gpu.download(&mut encoder);
        let sp = gpu.submit(&mut encoder);
        gpu.wait_for(&sp, !0);
        gpu.destroy_command_encoder(&mut encoder);

        // Read back to CPU
        let amps_f32 = state_gpu.read_amplitudes(gpu);
        let mut state = StateVec::new(num_qubits);
        for (i, &[re, im]) in amps_f32.iter().enumerate() {
            state.amplitudes[i] = num_complex::Complex64::new(re as f64, im as f64);
        }

        state_gpu.destroy(gpu);

        ExecutionResult {
            state,
            measurements,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_backend_runs_bell() {
        let mut circuit = Circuit::new(2);
        circuit.h(0).cnot(0, 1);
        let result = Backend::run_circuit(2, &circuit, None);
        let probs = result.state.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[3] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn backend_selects_cpu_for_small_circuits() {
        let backend = Backend::new(4, None);
        assert_eq!(backend.kind(), BackendKind::Cpu);
    }

    #[test]
    fn backend_threshold() {
        // Without GPU resources, always CPU
        let backend = Backend::new(20, None);
        assert_eq!(backend.kind(), BackendKind::Cpu);
    }
}
