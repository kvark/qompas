// Quantum state-vector compute shaders for blade-graphics.
//
// Complex numbers are stored as vec2<f32> where .x = real, .y = imag.
// The amplitude buffer has length 2^n, one vec2<f32> per basis state.

// ──────────────────────────────────────────────────────────────────────
// Common: complex multiplication
// ──────────────────────────────────────────────────────────────────────
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cadd(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return a + b;
}

// ──────────────────────────────────────────────────────────────────────
// Single-qubit gate
// ──────────────────────────────────────────────────────────────────────
// Push-constant params
struct Gate1Params {
    // 2x2 unitary as 4 complex numbers (row-major)
    m00: vec2<f32>,
    m01: vec2<f32>,
    m10: vec2<f32>,
    m11: vec2<f32>,
    // Bit mask for the target qubit: 1 << target
    target_bit: u32,
    // Total number of amplitude pairs to process: 2^(n-1)
    num_pairs: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> g1_params: Gate1Params;
@group(0) @binding(1) var<storage, read_write> g1_amps: array<vec2<f32>>;

@compute @workgroup_size(256)
fn apply_gate1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx = gid.x;
    if pair_idx >= g1_params.num_pairs {
        return;
    }

    // Map pair_idx to the index with the target bit = 0.
    // Insert a 0-bit at the target_bit position.
    let bit = g1_params.target_bit;
    let lo = pair_idx & (bit - 1u);
    let hi = (pair_idx >> findFirstBitHigh(bit)) << (findFirstBitHigh(bit) + 1u);
    let i0 = lo | hi;          // target bit = 0
    let i1 = i0 | bit;         // target bit = 1

    let a0 = g1_amps[i0];
    let a1 = g1_amps[i1];

    g1_amps[i0] = cadd(cmul(g1_params.m00, a0), cmul(g1_params.m01, a1));
    g1_amps[i1] = cadd(cmul(g1_params.m10, a0), cmul(g1_params.m11, a1));
}

// ──────────────────────────────────────────────────────────────────────
// Two-qubit gate
// ──────────────────────────────────────────────────────────────────────
struct Gate2Params {
    // 4x4 unitary as 16 complex numbers (row-major)
    m: array<vec2<f32>, 16>,
    bit0: u32,
    bit1: u32,
    num_quads: u32,  // 2^(n-2)
    _pad: u32,
}

@group(0) @binding(2) var<uniform> g2_params: Gate2Params;
@group(0) @binding(3) var<storage, read_write> g2_amps: array<vec2<f32>>;

// Map a quad_idx to the base index with both target bits cleared.
fn quad_base(quad_idx: u32, b0: u32, b1: u32) -> u32 {
    // We need to insert two zero bits at positions b0 and b1.
    let lo_bit = min(b0, b1);
    let hi_bit = max(b0, b1);
    let lo_shift = findFirstBitHigh(lo_bit);
    let hi_shift = findFirstBitHigh(hi_bit);

    // Extract three segments: below lo_bit, between, above hi_bit
    let seg0 = quad_idx & (lo_bit - 1u);
    let seg1 = ((quad_idx >> lo_shift) & ((hi_bit >> (lo_shift + 1u)) - 1u)) << (lo_shift + 1u);
    let seg2 = (quad_idx >> (lo_shift + hi_shift - lo_shift)) << (hi_shift + 1u);

    return seg0 | seg1 | seg2;
}

@compute @workgroup_size(256)
fn apply_gate2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let quad_idx = gid.x;
    if quad_idx >= g2_params.num_quads {
        return;
    }

    let b0 = g2_params.bit0;
    let b1 = g2_params.bit1;
    let base = quad_base(quad_idx, b0, b1);

    let i00 = base;
    let i01 = base | b1;
    let i10 = base | b0;
    let i11 = base | b0 | b1;

    let a = array<vec2<f32>, 4>(
        g2_amps[i00],
        g2_amps[i01],
        g2_amps[i10],
        g2_amps[i11],
    );

    // Matrix-vector multiply: out[row] = sum_j M[row*4+j] * a[j]
    for (var row = 0u; row < 4u; row++) {
        var acc = vec2<f32>(0.0, 0.0);
        for (var j = 0u; j < 4u; j++) {
            acc = cadd(acc, cmul(g2_params.m[row * 4u + j], a[j]));
        }
        let idx = select(
            select(select(i11, i10, row == 2u), i01, row == 1u),
            i00,
            row == 0u,
        );
        g2_amps[idx] = acc;
    }
}

// ──────────────────────────────────────────────────────────────────────
// Probability reduction (for measurement)
// ──────────────────────────────────────────────────────────────────────
struct MeasureParams {
    target_bit: u32,
    num_states: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(4) var<uniform> meas_params: MeasureParams;
@group(0) @binding(5) var<storage, read> meas_amps: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> meas_probs: array<f32>;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn measure_prob(@builtin(global_invocation_id) gid: vec3<u32>,
                @builtin(local_invocation_id) lid: vec3<u32>,
                @builtin(workgroup_id) wid: vec3<u32>) {
    let idx = gid.x;
    var val = 0.0f;
    if idx < meas_params.num_states {
        // Only sum amplitudes where target bit = 0
        if (idx & meas_params.target_bit) == 0u {
            let a = meas_amps[idx];
            val = a.x * a.x + a.y * a.y;  // |amplitude|^2
        }
    }

    // Workgroup reduction
    shared_sum[lid.x] = val;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if lid.x < stride {
            shared_sum[lid.x] += shared_sum[lid.x + stride];
        }
        workgroupBarrier();
    }

    if lid.x == 0u {
        meas_probs[wid.x] = shared_sum[0];
    }
}
