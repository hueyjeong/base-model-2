// SSD Stage 2: Chunk state accumulation
// chunk_states[c,h,p,n] = Σ_l decay(l) * dt[l] * x[l,h,p] * B[l,g,n]
// dispatch: (nchunks * nheads, 1, 1), workgroup_size: (64, 1, 1) — p 차원 병렬

@group(0) @binding(0) var<storage, read> x: array<f32>;        // [seq_len, d_inner]
@group(0) @binding(1) var<storage, read> B: array<f32>;        // [seq_len, ngroups * d_state]
@group(0) @binding(2) var<storage, read> dt: array<f32>;       // [seq_len, nheads]
@group(0) @binding(3) var<storage, read> dA_cumsum: array<f32>; // [nchunks, nheads, chunk_size]
@group(0) @binding(4) var<storage, read_write> chunk_states: array<f32>; // [nchunks, nheads, headdim, d_state]

struct Params {
    seq_len: u32,
    nheads: u32,
    headdim: u32,
    d_state: u32,
    ngroups: u32,
    chunk_size: u32,
    nchunks: u32,
    d_inner: u32,
};
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let wg = gid.x / 64u;
    if (wg >= params.nchunks * params.nheads) { return; }

    let c = wg / params.nheads;
    let h = wg % params.nheads;
    let p = lid;  // headdim index (0..63)
    if (p >= params.headdim) { return; }

    let heads_per_group = params.nheads / params.ngroups;
    let g = h / heads_per_group;

    let dA_last = dA_cumsum[(c * params.nheads + h) * params.chunk_size + params.chunk_size - 1u];

    for (var n: u32 = 0u; n < params.d_state; n++) {
        var acc: f32 = 0.0;
        for (var l: u32 = 0u; l < params.chunk_size; l++) {
            let t = c * params.chunk_size + l;
            if (t >= params.seq_len) { break; }

            let dA_l = dA_cumsum[(c * params.nheads + h) * params.chunk_size + l];
            let decay = exp(dA_last - dA_l);
            let dt_val = dt[t * params.nheads + h];
            let x_val = x[t * params.d_inner + h * params.headdim + p];
            let b_val = B[t * params.ngroups * params.d_state + g * params.d_state + n];

            acc += b_val * x_val * decay * dt_val;
        }
        chunk_states[((c * params.nheads + h) * params.headdim + p) * params.d_state + n] = acc;
    }
}
