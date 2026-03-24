// Embedding lookup: out[t, d] = embedding[ids[t], d] * scale

@group(0) @binding(0) var<storage, read> embedding: array<f32>;
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Params {
    d_model: u32,
    seq_len: u32,
    scale: f32,
};
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.seq_len * params.d_model;
    if (idx >= total) { return; }

    let t = idx / params.d_model;
    let d = idx % params.d_model;
    let token_id = ids[t];
    out[idx] = embedding[token_id * params.d_model + d] * params.scale;
}
