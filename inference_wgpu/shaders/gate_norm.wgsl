// Gate + RMSNorm 퓨전: y[t,d] = RMSNorm(y_scan[t,d] * SiLU(z[t,d]), weight)
// z는 proj 버퍼의 첫 d_inner 원소
// 1 워크그룹 = 1 토큰

@group(0) @binding(0) var<storage, read_write> y_scan: array<f32>;  // [seq_len, d_inner] — in-place
@group(0) @binding(1) var<storage, read> z: array<f32>;             // [seq_len, d_in_proj] (z = 첫 d_inner)
@group(0) @binding(2) var<storage, read> weight: array<f32>;        // [d_inner]

struct Params {
    d_inner: u32,
    d_in_proj: u32,  // z 버퍼의 stride (d_in_proj)
    eps: f32,
};
var<push_constant> params: Params;

fn silu(v: f32) -> f32 {
    return v / (1.0 + exp(-v));
}

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let t = wid.x;
    let y_base = t * params.d_inner;
    let z_base = t * params.d_in_proj;  // z는 proj의 처음 d_inner

    // Phase 1: gate + sum of squares
    var local_sq: f32 = 0.0;
    for (var i = lid; i < params.d_inner; i += 256u) {
        let gated = y_scan[y_base + i] * silu(z[z_base + i]);
        y_scan[y_base + i] = gated;
        local_sq += gated * gated;
    }
    shared_sum[lid] = local_sq;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            shared_sum[lid] += shared_sum[lid + stride];
        }
        workgroupBarrier();
    }

    let rms_inv = inverseSqrt(shared_sum[0] / f32(params.d_inner) + params.eps);

    // Phase 2: normalize + scale
    for (var i = lid; i < params.d_inner; i += 256u) {
        y_scan[y_base + i] = y_scan[y_base + i] * rms_inv * weight[i];
    }
}
