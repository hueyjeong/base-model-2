// SwiGLU (ReGLU): out[t, i] = relu(gate[t, i]) * up[t, i]
// 입력: gu[seq, 2*d_ff] — 전반부 gate, 후반부 up
// 출력: out[seq, d_ff]

@group(0) @binding(0) var<storage, read> gu: array<f32>;           // [seq, 2*d_ff]
@group(0) @binding(1) var<storage, read_write> out: array<f32>;    // [seq, d_ff]

struct Params {
    seq_len: u32,
    d_ff: u32,
};
var<push_constant> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.seq_len * params.d_ff;
    if (idx >= total) { return; }

    let t = idx / params.d_ff;
    let i = idx % params.d_ff;
    let dff2 = params.d_ff * 2u;

    let gate = gu[t * dff2 + i];
    let up = gu[t * dff2 + params.d_ff + i];

    // ReLU(gate) * up
    out[idx] = max(gate, 0.0) * up;
}
