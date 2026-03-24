// RMSNorm: out[t,d] = x[t,d] * rsqrt(mean(x[t,:]²) + eps) * weight[d]
// 1 워크그룹 = 1 토큰

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Params {
    d: u32,
    eps: f32,
};
var<push_constant> params: Params;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let t = wid.x;
    let base = t * params.d;

    // Phase 1: sum of squares
    var local_sq: f32 = 0.0;
    for (var i = lid; i < params.d; i += 256u) {
        let v = x[base + i];
        local_sq += v * v;
    }
    shared_sum[lid] = local_sq;
    workgroupBarrier();

    // Tree reduction
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            shared_sum[lid] += shared_sum[lid + stride];
        }
        workgroupBarrier();
    }

    let rms_inv = inverseSqrt(shared_sum[0] / f32(params.d) + params.eps);

    // Phase 2: normalize + scale
    for (var i = lid; i < params.d; i += 256u) {
        out[base + i] = x[base + i] * rms_inv * weight[i];
    }
}
