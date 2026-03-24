// 활성화 함수: SiLU, Softplus (element-wise)

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

struct Params {
    n: u32,          // 총 원소 수
    act_type: u32,   // 0=SiLU, 1=Softplus
};
@group(0) @binding(2) var<uniform> params: Params;

fn silu(v: f32) -> f32 {
    return v / (1.0 + exp(-v));
}

fn softplus(v: f32) -> f32 {
    // ln(1 + exp(v)), 수치 안정
    if (v > 20.0) { return v; }
    return log(1.0 + exp(v));
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) { return; }

    let v = x[idx];
    if (params.act_type == 0u) {
        out[idx] = silu(v);
    } else {
        out[idx] = softplus(v);
    }
}
