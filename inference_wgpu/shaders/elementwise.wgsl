// Element-wise 연산: residual_add, add_inplace, residual_add3

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Params {
    n: u32,
};
var<push_constant> params: Params;

// out[i] = a[i] + b[i]
@compute @workgroup_size(256, 1, 1)
fn residual_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) { return; }
    out[idx] = a[idx] + b[idx];
}

// out[i] += a[i] (in-place addition, a는 read, out는 read_write)
@compute @workgroup_size(256, 1, 1)
fn add_inplace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) { return; }
    out[idx] += a[idx];
}

// out[i] = a[i] + b[i] + out[i] (3-way residual, out에 기존 값이 있어야 함)
@compute @workgroup_size(256, 1, 1)
fn residual_add3(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) { return; }
    out[idx] += a[idx] + b[idx];
}
