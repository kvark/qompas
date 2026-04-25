//! GPU-accelerated state-vector simulator using blade-graphics compute shaders.
//!
//! [`StateGpu`] stores the 2^n amplitude vector in a GPU storage buffer and
//! applies gates via compute shader dispatches. This gives massive parallelism
//! for large qubit counts where the CPU path becomes a bottleneck.
//!
//! Complex amplitudes use `f32` pairs on the GPU (vs `f64` on CPU) for
//! throughput — this is the standard trade-off in GPU quantum simulation.

use crate::gates::{Gate1, Gate2};

/// Push-constant layout for single-qubit gate dispatch.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Gate1Params {
    pub m00: [f32; 2],
    pub m01: [f32; 2],
    pub m10: [f32; 2],
    pub m11: [f32; 2],
    pub target_bit: u32,
    pub num_pairs: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Push-constant layout for two-qubit gate dispatch.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Gate2Params {
    pub m: [[f32; 2]; 16],
    pub bit0: u32,
    pub bit1: u32,
    pub num_quads: u32,
    pub _pad: u32,
}

/// Push-constant layout for measurement probability reduction.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct MeasureParams {
    pub target_bit: u32,
    pub num_states: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Shader data binding for single-qubit gate.
#[derive(blade_macros::ShaderData)]
pub struct Gate1Data {
    pub g1_params: Gate1Params,
    pub g1_amps: blade_graphics::BufferPiece,
}

/// Shader data binding for two-qubit gate.
#[derive(blade_macros::ShaderData)]
pub struct Gate2Data {
    pub g2_params: Gate2Params,
    pub g2_amps: blade_graphics::BufferPiece,
}

/// Shader data binding for measurement.
#[derive(blade_macros::ShaderData)]
pub struct MeasureData {
    pub meas_params: MeasureParams,
    pub meas_amps: blade_graphics::BufferPiece,
    pub meas_probs: blade_graphics::BufferPiece,
}

const WORKGROUP_SIZE: u32 = 256;
const SHADER_SOURCE: &str = include_str!("shaders/quantum.wgsl");

/// GPU compute pipelines for quantum operations.
pub struct QuantumPipelines {
    gate1_pipeline: blade_graphics::ComputePipeline,
    gate2_pipeline: blade_graphics::ComputePipeline,
    measure_pipeline: blade_graphics::ComputePipeline,
}

impl QuantumPipelines {
    /// Create all quantum compute pipelines.
    pub fn new(gpu: &blade_graphics::Context) -> Self {
        let shader = gpu.create_shader(blade_graphics::ShaderDesc {
            source: SHADER_SOURCE,
        });

        let gate1_layout = <Gate1Data as blade_graphics::ShaderData>::layout();
        let gate1_pipeline = gpu.create_compute_pipeline(blade_graphics::ComputePipelineDesc {
            name: "gate1",
            data_layouts: &[&gate1_layout],
            compute: blade_graphics::ShaderFunction {
                shader: &shader,
                entry_point: "apply_gate1",
            },
        });

        let gate2_layout = <Gate2Data as blade_graphics::ShaderData>::layout();
        let gate2_pipeline = gpu.create_compute_pipeline(blade_graphics::ComputePipelineDesc {
            name: "gate2",
            data_layouts: &[&gate2_layout],
            compute: blade_graphics::ShaderFunction {
                shader: &shader,
                entry_point: "apply_gate2",
            },
        });

        let measure_layout = <MeasureData as blade_graphics::ShaderData>::layout();
        let measure_pipeline = gpu.create_compute_pipeline(blade_graphics::ComputePipelineDesc {
            name: "measure_prob",
            data_layouts: &[&measure_layout],
            compute: blade_graphics::ShaderFunction {
                shader: &shader,
                entry_point: "measure_prob",
            },
        });

        Self {
            gate1_pipeline,
            gate2_pipeline,
            measure_pipeline,
        }
    }

    pub fn destroy(&mut self, gpu: &blade_graphics::Context) {
        gpu.destroy_compute_pipeline(&mut self.gate1_pipeline);
        gpu.destroy_compute_pipeline(&mut self.gate2_pipeline);
        gpu.destroy_compute_pipeline(&mut self.measure_pipeline);
    }
}

/// GPU-backed quantum state vector.
///
/// Amplitudes live in a GPU storage buffer. Gate application and measurement
/// are dispatched as compute shaders for massive parallelism.
pub struct StateGpu {
    pub num_qubits: usize,
    /// GPU storage buffer: 2^n × vec2<f32> (8 bytes per amplitude).
    amp_buffer: blade_graphics::Buffer,
    /// Staging buffer for CPU↔GPU transfers (shared memory).
    staging_buffer: blade_graphics::Buffer,
    /// Reduction output buffer for measurement.
    reduce_buffer: blade_graphics::Buffer,
    /// Staging for reading back reduction results.
    reduce_staging: blade_graphics::Buffer,
    /// Number of workgroups needed for full reduction.
    reduce_groups: u32,
}

impl StateGpu {
    /// Allocate a GPU state vector for `n` qubits, initialised to |0…0⟩.
    pub fn new(num_qubits: usize, gpu: &blade_graphics::Context) -> Self {
        let dim = 1u64 << num_qubits;
        let amp_bytes = dim * 8; // 2 × f32 per amplitude

        let amp_buffer = gpu.create_buffer(blade_graphics::BufferDesc {
            name: "amplitudes",
            size: amp_bytes,
            memory: blade_graphics::Memory::Device,
        });

        let staging_buffer = gpu.create_buffer(blade_graphics::BufferDesc {
            name: "staging",
            size: amp_bytes,
            memory: blade_graphics::Memory::Shared,
        });

        let reduce_groups = dim.div_ceil(WORKGROUP_SIZE as u64) as u32;
        let reduce_bytes = reduce_groups as u64 * 4; // f32 per workgroup

        let reduce_buffer = gpu.create_buffer(blade_graphics::BufferDesc {
            name: "reduce",
            size: reduce_bytes,
            memory: blade_graphics::Memory::Device,
        });

        let reduce_staging = gpu.create_buffer(blade_graphics::BufferDesc {
            name: "reduce_staging",
            size: reduce_bytes,
            memory: blade_graphics::Memory::Shared,
        });

        // Initialize |0…0⟩ in staging buffer, then upload.
        unsafe {
            let ptr = staging_buffer.data() as *mut [f32; 2];
            // Zero everything
            std::ptr::write_bytes(ptr, 0, dim as usize);
            // Set amplitude[0] = (1.0, 0.0)
            *ptr = [1.0f32, 0.0f32];
        }
        gpu.sync_buffer(staging_buffer);

        StateGpu {
            num_qubits,
            amp_buffer,
            staging_buffer,
            reduce_buffer,
            reduce_staging,
            reduce_groups,
        }
    }

    /// Upload initial state from staging → device.
    pub fn upload_initial(&self, encoder: &mut blade_graphics::CommandEncoder) {
        let dim = 1u64 << self.num_qubits;
        let size = dim * 8;
        let mut transfer = encoder.transfer("upload_amps");
        transfer.copy_buffer_to_buffer(self.staging_buffer.into(), self.amp_buffer.into(), size);
    }

    /// Download amplitudes from device → staging for CPU readback.
    pub fn download(&self, encoder: &mut blade_graphics::CommandEncoder) {
        let dim = 1u64 << self.num_qubits;
        let size = dim * 8;
        let mut transfer = encoder.transfer("download_amps");
        transfer.copy_buffer_to_buffer(self.amp_buffer.into(), self.staging_buffer.into(), size);
    }

    /// Get the raw pointer to the staging buffer for direct writes.
    ///
    /// # Safety
    /// Caller must ensure writes don't exceed buffer size and that
    /// the buffer is not concurrently accessed by the GPU.
    pub unsafe fn staging_buffer_ptr(&self) -> *mut u8 {
        self.staging_buffer.data()
    }

    /// Read amplitudes back to CPU (call after download + submit + wait).
    pub fn read_amplitudes(&self, gpu: &blade_graphics::Context) -> Vec<[f32; 2]> {
        gpu.sync_buffer(self.staging_buffer);
        let dim = 1usize << self.num_qubits;
        let mut result = vec![[0.0f32; 2]; dim];
        unsafe {
            let ptr = self.staging_buffer.data() as *const [f32; 2];
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), dim);
        }
        result
    }

    /// Dispatch a single-qubit gate.
    pub fn dispatch_gate1(
        &self,
        encoder: &mut blade_graphics::CommandEncoder,
        pipelines: &QuantumPipelines,
        gate: &Gate1,
        target: usize,
    ) {
        let num_pairs = (1u32 << self.num_qubits) >> 1;
        let groups = num_pairs.div_ceil(WORKGROUP_SIZE);

        let m = &gate.matrix;
        let params = Gate1Params {
            m00: [m[(0, 0)].re as f32, m[(0, 0)].im as f32],
            m01: [m[(0, 1)].re as f32, m[(0, 1)].im as f32],
            m10: [m[(1, 0)].re as f32, m[(1, 0)].im as f32],
            m11: [m[(1, 1)].re as f32, m[(1, 1)].im as f32],
            target_bit: 1u32 << target,
            num_pairs,
            _pad0: 0,
            _pad1: 0,
        };

        let mut compute = encoder.compute("gate1");
        let mut pe = compute.with(&pipelines.gate1_pipeline);
        pe.bind(
            0,
            &Gate1Data {
                g1_params: params,
                g1_amps: self.amp_buffer.into(),
            },
        );
        pe.dispatch([groups, 1, 1]);
    }

    /// Dispatch a two-qubit gate.
    pub fn dispatch_gate2(
        &self,
        encoder: &mut blade_graphics::CommandEncoder,
        pipelines: &QuantumPipelines,
        gate: &Gate2,
        qubit0: usize,
        qubit1: usize,
    ) {
        let num_quads = (1u32 << self.num_qubits) >> 2;
        let groups = num_quads.div_ceil(WORKGROUP_SIZE);

        let mut m = [[0.0f32; 2]; 16];
        for (i, slot) in m.iter_mut().enumerate() {
            *slot = [gate.matrix[i].re as f32, gate.matrix[i].im as f32];
        }

        let params = Gate2Params {
            m,
            bit0: 1u32 << qubit0,
            bit1: 1u32 << qubit1,
            num_quads,
            _pad: 0,
        };

        let mut compute = encoder.compute("gate2");
        let mut pe = compute.with(&pipelines.gate2_pipeline);
        pe.bind(
            0,
            &Gate2Data {
                g2_params: params,
                g2_amps: self.amp_buffer.into(),
            },
        );
        pe.dispatch([groups, 1, 1]);
    }

    /// Dispatch measurement probability reduction.
    ///
    /// After submit + wait, read back via `read_prob0()`.
    pub fn dispatch_measure_prob(
        &self,
        encoder: &mut blade_graphics::CommandEncoder,
        pipelines: &QuantumPipelines,
        target: usize,
    ) {
        let num_states = 1u32 << self.num_qubits;
        let groups = num_states.div_ceil(WORKGROUP_SIZE);

        let params = MeasureParams {
            target_bit: 1u32 << target,
            num_states,
            _pad0: 0,
            _pad1: 0,
        };

        {
            let mut compute = encoder.compute("measure_prob");
            let mut pe = compute.with(&pipelines.measure_pipeline);
            pe.bind(
                0,
                &MeasureData {
                    meas_params: params,
                    meas_amps: self.amp_buffer.into(),
                    meas_probs: self.reduce_buffer.into(),
                },
            );
            pe.dispatch([groups, 1, 1]);
        }

        // Copy reduction results to staging for CPU readback
        let reduce_bytes = self.reduce_groups as u64 * 4;
        let mut transfer = encoder.transfer("download_probs");
        transfer.copy_buffer_to_buffer(
            self.reduce_buffer.into(),
            self.reduce_staging.into(),
            reduce_bytes,
        );
    }

    /// Read back prob(|0⟩) after measure dispatch + submit + wait.
    pub fn read_prob0(&self, gpu: &blade_graphics::Context) -> f32 {
        gpu.sync_buffer(self.reduce_staging);
        let mut sum = 0.0f32;
        unsafe {
            let ptr = self.reduce_staging.data() as *const f32;
            for i in 0..self.reduce_groups as usize {
                sum += *ptr.add(i);
            }
        }
        sum
    }

    /// Clean up GPU resources.
    pub fn destroy(&self, gpu: &blade_graphics::Context) {
        gpu.destroy_buffer(self.amp_buffer);
        gpu.destroy_buffer(self.staging_buffer);
        gpu.destroy_buffer(self.reduce_buffer);
        gpu.destroy_buffer(self.reduce_staging);
    }
}
