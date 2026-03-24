// Depthwise causal conv1d + SiLU + x/B/C split (퓨전)
//
// 입력: xbc[seq, d_conv_in]
// 출력: x_out[seq, d_inner], b_out[seq, ng*ds], c_out[seq, ng*ds]
//
// 각 출력 원소: SiLU( bias[ch] + Σ_{ki} weight[ch,ki] * xbc[t-(k-1)+ki, ch] )

@group(0) @binding(0) var<storage, read> xbc: array<f32>;        // [seq, d_conv_in]
@group(0) @binding(1) var<storage, read> weight: array<f32>;     // [d_conv_in, d_conv]
@group(0) @binding(2) var<storage, read> bias: array<f32>;       // [d_conv_in]
@group(0) @binding(3) var<storage, read_write> x_out: array<f32>; // [seq, d_inner]
@group(0) @binding(4) var<storage, read_write> b_out: array<f32>; // [seq, ng_ds]
@group(0) @binding(5) var<storage, read_write> c_out: array<f32>; // [seq, ng_ds]

struct Params {
    seq_len: u32,
    d_conv_in: u32,
    d_conv: u32,
    d_inner: u32,
    ng_ds: u32,
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
    let ch = idx % params.d_conv_in;

    // Causal conv1d
    var sum: f32 = bias[ch];
    for (var ki: u32 = 0u; ki < params.d_conv; ki++) {
        let src_t = i32(t) - i32(params.d_conv - 1u) + i32(ki);
        if (src_t >= 0 && u32(src_t) < params.seq_len) {
            sum += weight[ch * params.d_conv + ki] * xbc[u32(src_t) * params.d_conv_in + ch];
        }
    }

    // SiLU + split
    let val = silu(sum);
    if (ch < params.d_inner) {
        x_out[t * params.d_inner + ch] = val;
    } else if (ch < params.d_inner + params.ng_ds) {
        let bd = ch - params.d_inner;
        b_out[t * params.ng_ds + bd] = val;
    } else {
        let cd = ch - params.d_inner - params.ng_ds;
        c_out[t * params.ng_ds + cd] = val;
    }
}
