// Depthwise causal conv1d + SiLU
// out[t, ch] = SiLU( bias[ch] + Σ_{ki=0..d_conv-1} weight[ch, ki] * x[t - (d_conv-1) + ki, ch] )

@group(0) @binding(0) var<storage, read> x: array<f32>;       // [seq_len, channels]
@group(0) @binding(1) var<storage, read> weight: array<f32>;   // [channels, d_conv]
@group(0) @binding(2) var<storage, read> bias: array<f32>;     // [channels]
@group(0) @binding(3) var<storage, read_write> out: array<f32>; // [seq_len, channels]

struct Params {
    seq_len: u32,
    channels: u32,
    d_conv: u32,
};
@group(0) @binding(4) var<uniform> params: Params;

fn silu(v: f32) -> f32 {
    return v / (1.0 + exp(-v));
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.seq_len * params.channels;
    if (idx >= total) { return; }

    let t = idx / params.channels;
    let ch = idx % params.channels;

    var sum: f32 = bias[ch];
    for (var ki: u32 = 0u; ki < params.d_conv; ki++) {
        let src_t = i32(t) - i32(params.d_conv - 1u) + i32(ki);
        if (src_t >= 0 && u32(src_t) < params.seq_len) {
            sum += weight[ch * params.d_conv + ki] * x[u32(src_t) * params.channels + ch];
        }
    }

    out[idx] = silu(sum);
}
