// conv1d 출력에 SiLU 적용 후 x_conv, B_conv, C_conv로 분리
// 입력: xbc_conv[seq, d_conv_in] (conv1d 출력)
// 출력: x[seq, d_inner], B[seq, ngroups*d_state], C[seq, ngroups*d_state]
//
// 레이아웃: xbc_conv = [x(d_inner) | B(ng*ds) | C(ng*ds)]

@group(0) @binding(0) var<storage, read> xbc_conv: array<f32>;     // [seq, d_conv_in]
@group(0) @binding(1) var<storage, read_write> x_out: array<f32>;  // [seq, d_inner]
@group(0) @binding(2) var<storage, read_write> b_out: array<f32>;  // [seq, ng*ds]
@group(0) @binding(3) var<storage, read_write> c_out: array<f32>;  // [seq, ng*ds]

struct Params {
    seq_len: u32,
    d_inner: u32,
    d_conv_in: u32,
    ng_ds: u32,       // ngroups * d_state
};
var<push_constant> params: Params;

fn silu(v: f32) -> f32 {
    return v / (1.0 + exp(-v));
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.seq_len * params.d_conv_in;
    if (idx >= total) { return; }

    let t = idx / params.d_conv_in;
    let d = idx % params.d_conv_in;
    let val = silu(xbc_conv[idx]);

    if (d < params.d_inner) {
        // x 부분
        x_out[t * params.d_inner + d] = val;
    } else if (d < params.d_inner + params.ng_ds) {
        // B 부분
        let bd = d - params.d_inner;
        b_out[t * params.ng_ds + bd] = val;
    } else {
        // C 부분
        let cd = d - params.d_inner - params.ng_ds;
        c_out[t * params.ng_ds + cd] = val;
    }
}
