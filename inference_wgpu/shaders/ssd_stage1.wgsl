// SSD Stage 1: dA cumulative sum
// 각 (chunk, head) 쌍에 대해 독립적으로 cumsum 수행
// dA_cumsum[c,h,l] = Σ_{i=0..l} A[h] * dt[c*chunk_size+i, h]

@group(0) @binding(0) var<storage, read> dt: array<f32>;      // [seq_len, nheads]
@group(0) @binding(1) var<storage, read> a_neg: array<f32>;    // [nheads]
@group(0) @binding(2) var<storage, read_write> dA_cumsum: array<f32>; // [nchunks, nheads, chunk_size]

struct Params {
    seq_len: u32,
    nheads: u32,
    chunk_size: u32,
    nchunks: u32,
};
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ch = gid.x;
    if (ch >= params.nchunks * params.nheads) { return; }

    let c = ch / params.nheads;
    let h = ch % params.nheads;

    var cumsum: f32 = 0.0;
    for (var l: u32 = 0u; l < params.chunk_size; l++) {
        let t = c * params.chunk_size + l;
        var dt_val: f32 = 0.0;
        if (t < params.seq_len) {
            dt_val = dt[t * params.nheads + h];
        }
        cumsum += a_neg[h] * dt_val;
        dA_cumsum[(c * params.nheads + h) * params.chunk_size + l] = cumsum;
    }
}
