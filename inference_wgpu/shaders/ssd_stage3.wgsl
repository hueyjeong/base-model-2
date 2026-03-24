// SSD Stage 3: Inter-chunk state propagation (순차)
// running_state[h,p,n] = decay * running_state[h,p,n] + chunk_states[c,h,p,n]
// dispatch: (nheads, 1, 1), workgroup_size: (64, 1, 1) — p 차원 병렬

@group(0) @binding(0) var<storage, read> chunk_states: array<f32>;   // [nchunks, nheads, headdim, d_state]
@group(0) @binding(1) var<storage, read> dA_cumsum: array<f32>;      // [nchunks, nheads, chunk_size]
@group(0) @binding(2) var<storage, read_write> prev_states: array<f32>; // [nchunks, nheads, headdim, d_state]

struct Params {
    nheads: u32,
    headdim: u32,
    d_state: u32,
    chunk_size: u32,
    nchunks: u32,
};
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let h = wid.x;
    if (h >= params.nheads) { return; }
    let p = lid;
    if (p >= params.headdim) { return; }

    // running state: private memory
    var running: array<f32, 64>;  // d_state=64
    for (var n: u32 = 0u; n < params.d_state; n++) {
        running[n] = 0.0;
    }

    for (var c: u32 = 0u; c < params.nchunks; c++) {
        // prev_states[c, h, p, :] = running[:]
        for (var n: u32 = 0u; n < params.d_state; n++) {
            prev_states[((c * params.nheads + h) * params.headdim + p) * params.d_state + n] = running[n];
        }

        // decay + accumulate
        let dA_total = dA_cumsum[(c * params.nheads + h) * params.chunk_size + params.chunk_size - 1u];
        let inter_decay = exp(dA_total);
        for (var n: u32 = 0u; n < params.d_state; n++) {
            let idx = ((c * params.nheads + h) * params.headdim + p) * params.d_state + n;
            running[n] = inter_decay * running[n] + chunk_states[idx];
        }
    }
}
