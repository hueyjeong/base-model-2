// SSD Stage 4: Final output computation
// y[t, h*headdim+p] = intra_chunk + inter_chunk + D[h] * x[t, h*headdim+p]
// dispatch: (nheads * nchunks, 1, 1), workgroup_size: (64, 1, 1) — p 차원 병렬

@group(0) @binding(0) var<storage, read> x: array<f32>;           // [seq_len, d_inner]
@group(0) @binding(1) var<storage, read> B: array<f32>;           // [seq_len, ngroups * d_state]
@group(0) @binding(2) var<storage, read> C: array<f32>;           // [seq_len, ngroups * d_state]
@group(0) @binding(3) var<storage, read> dt: array<f32>;          // [seq_len, nheads]
@group(0) @binding(4) var<storage, read> dA_cumsum: array<f32>;   // [nchunks, nheads, chunk_size]
@group(0) @binding(5) var<storage, read> prev_states: array<f32>; // [nchunks, nheads, headdim, d_state]
@group(0) @binding(6) var<storage, read> D: array<f32>;           // [nheads]
@group(0) @binding(7) var<storage, read_write> y: array<f32>;     // [seq_len, d_inner]

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
@group(0) @binding(8) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let wg = gid.x / 64u;
    if (wg >= params.nheads * params.nchunks) { return; }

    let h = wg / params.nchunks;
    let c = wg % params.nchunks;
    let p = lid;
    if (p >= params.headdim) { return; }

    let heads_per_group = params.nheads / params.ngroups;
    let g = h / heads_per_group;

    for (var l: u32 = 0u; l < params.chunk_size; l++) {
        let t = c * params.chunk_size + l;
        if (t >= params.seq_len) { break; }

        let dA_l = dA_cumsum[(c * params.nheads + h) * params.chunk_size + l];

        // Intra-chunk contribution: Σ_{s≤l} CB[l,s] * decay * dt_s * x[s,h,p]
        var intra: f32 = 0.0;
        for (var s: u32 = 0u; s <= l; s++) {
            let t_s = c * params.chunk_size + s;
            if (t_s >= params.seq_len) { break; }

            // CB dot product
            var cb: f32 = 0.0;
            for (var n: u32 = 0u; n < params.d_state; n++) {
                cb += C[t * params.ngroups * params.d_state + g * params.d_state + n]
                    * B[t_s * params.ngroups * params.d_state + g * params.d_state + n];
            }

            let dA_s = dA_cumsum[(c * params.nheads + h) * params.chunk_size + s];
            let decay = exp(dA_l - dA_s);
            let dt_s = dt[t_s * params.nheads + h];
            intra += cb * decay * dt_s * x[t_s * params.d_inner + h * params.headdim + p];
        }

        // Inter-chunk contribution: Σ_n C[t,g,n] * prev_states[c,h,p,n] * decay_from_0
        let state_decay = exp(dA_l);
        var inter: f32 = 0.0;
        for (var n: u32 = 0u; n < params.d_state; n++) {
            inter += C[t * params.ngroups * params.d_state + g * params.d_state + n]
                   * prev_states[((c * params.nheads + h) * params.headdim + p) * params.d_state + n];
        }
        inter *= state_decay;

        // Skip connection
        let skip = D[h] * x[t * params.d_inner + h * params.headdim + p];

        y[t * params.d_inner + h * params.headdim + p] = intra + inter + skip;
    }
}
