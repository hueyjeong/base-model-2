// proj 버퍼에서 xBC와 dt를 추출
// proj 레이아웃: [z(d_inner) | xBC(d_conv_in) | dt(nheads)] per token
//
// 출력 1: xbc[seq, d_conv_in] = proj[:, d_inner : d_inner+d_conv_in]
// 출력 2: dt[seq, nheads] = softplus(proj[:, d_inner+d_conv_in :] + dt_bias)

@group(0) @binding(0) var<storage, read> proj: array<f32>;       // [seq, d_in_proj]
@group(0) @binding(1) var<storage, read> dt_bias: array<f32>;    // [nheads]
@group(0) @binding(2) var<storage, read_write> xbc: array<f32>;  // [seq, d_conv_in]
@group(0) @binding(3) var<storage, read_write> dt_out: array<f32>; // [seq, nheads]

struct Params {
    seq_len: u32,
    d_in_proj: u32,
    d_inner: u32,
    d_conv_in: u32,
    nheads: u32,
};
var<push_constant> params: Params;

fn softplus(v: f32) -> f32 {
    if (v > 20.0) { return v; }
    return log(1.0 + exp(v));
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total_xbc = params.seq_len * params.d_conv_in;
    let total_dt = params.seq_len * params.nheads;

    // xBC 추출
    if (idx < total_xbc) {
        let t = idx / params.d_conv_in;
        let d = idx % params.d_conv_in;
        xbc[idx] = proj[t * params.d_in_proj + params.d_inner + d];
    }

    // dt 추출 + softplus + bias
    if (idx < total_dt) {
        let t = idx / params.nheads;
        let h = idx % params.nheads;
        let raw = proj[t * params.d_in_proj + params.d_inner + params.d_conv_in + h];
        dt_out[idx] = softplus(raw + dt_bias[h]);
    }
}
