// 시퀀스 뒤집기: out[t, d] = x[seq_len-1-t, d]

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

struct Params {
    d_model: u32,
    seq_len: u32,
};
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.seq_len * params.d_model;
    if (idx >= total) { return; }

    let t = idx / params.d_model;
    let d = idx % params.d_model;
    let rev_t = params.seq_len - 1u - t;
    out[rev_t * params.d_model + d] = x[idx];
}
