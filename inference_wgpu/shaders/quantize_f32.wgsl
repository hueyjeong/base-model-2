// BitLinear용 per-token float quantization
// 입력: x[seq, d] (LayerNorm 완료된 상태)
// 출력: out[seq, d] = round(x * 127 / eta), scales[seq] = eta / 127
// 여기서 eta = max(|x[t,:]|)
//
// 1 워크그룹 = 1 토큰

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read_write> scales: array<f32>;

struct Params {
    d: u32,
};
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> shared_max: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let t = wid.x;
    let base = t * params.d;

    // Phase 1: max(|x|)
    var local_max: f32 = 0.0;
    for (var i = lid; i < params.d; i += 256u) {
        local_max = max(local_max, abs(x[base + i]));
    }
    shared_max[lid] = local_max;
    workgroupBarrier();

    // Tree reduction
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            shared_max[lid] = max(shared_max[lid], shared_max[lid + stride]);
        }
        workgroupBarrier();
    }

    let eta = max(shared_max[0], 1e-5);
    let inv_eta = 127.0 / eta;

    // Scale 저장
    if (lid == 0u) {
        scales[t] = eta / 127.0;
    }

    // Phase 2: quantize
    for (var i = lid; i < params.d; i += 256u) {
        out[base + i] = round(clamp(x[base + i] * inv_eta, -128.0, 127.0));
    }
}
