// Per-token argmax: tags[t] = argmax_j(logits[t, j])
// 256-thread parallel reduction

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> tags: array<u32>;

struct Params {
    n_tags: u32,
    seq_len: u32,
};
var<push_constant> params: Params;

var<workgroup> smem_val: array<f32, 256>;
var<workgroup> smem_idx: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let t = wid.x;
    if (t >= params.seq_len) { return; }

    let base = t * params.n_tags;

    // Phase 1: strided local max (각 스레드가 n_tags/256개씩 분담)
    var best_val: f32 = -1e30;
    var best_idx: u32 = 0u;
    for (var j = lid; j < params.n_tags; j += 256u) {
        let v = logits[base + j];
        if (v > best_val) {
            best_val = v;
            best_idx = j;
        }
    }
    smem_val[lid] = best_val;
    smem_idx[lid] = best_idx;
    workgroupBarrier();

    // Phase 2: tree reduction
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) {
            if (smem_val[lid + s] > smem_val[lid]) {
                smem_val[lid] = smem_val[lid + s];
                smem_idx[lid] = smem_idx[lid + s];
            }
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        tags[t] = smem_idx[0];
    }
}
