// Per-token argmax: tags[t] = argmax_j(logits[t, j])

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> tags: array<u32>;

struct Params {
    n_tags: u32,
    seq_len: u32,
};
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= params.seq_len) { return; }

    let base = t * params.n_tags;
    var best_idx: u32 = 0u;
    var best_val: f32 = logits[base];

    for (var j = 1u; j < params.n_tags; j++) {
        let v = logits[base + j];
        if (v > best_val) {
            best_val = v;
            best_idx = j;
        }
    }

    tags[t] = best_idx;
}
