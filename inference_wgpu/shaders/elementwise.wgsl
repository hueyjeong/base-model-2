// Element-wise 연산: residual_add, residual_add3

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Params {
    n: u32,
};
@group(0) @binding(3) var<uniform> params: Params;

// out[i] = a[i] + b[i]
@compute @workgroup_size(256, 1, 1)
fn residual_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) { return; }
    out[idx] = a[idx] + b[idx];
}
