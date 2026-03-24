// SSD Stage 4: Final output computation
// y[t, h*headdim+p] = intra_chunk + inter_chunk + D[h] * x[t, h*headdim+p]
//
// 최적화: barrier 없이 각 스레드가 CB dot product를 직접 계산.
// d_state=64 루프는 GPU에서 ~2ns, barrier는 ~100ns이므로 직접 계산이 빠름.
// 워크그룹을 (h, chunk_l) 단위로 할당하여 GPU 병렬성 극대화.
//
// dispatch: (nheads * nchunks * chunk_size, 1, 1), workgroup_size: (64, 1, 1)
// 각 워크그룹: 특정 (h, c, l)의 p=0..63 출력 계산

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read> C: array<f32>;
@group(0) @binding(3) var<storage, read> dt: array<f32>;
@group(0) @binding(4) var<storage, read> dA_cumsum: array<f32>;
@group(0) @binding(5) var<storage, read> prev_states: array<f32>;
@group(0) @binding(6) var<storage, read> D: array<f32>;
@group(0) @binding(7) var<storage, read_write> y: array<f32>;

struct Params {
    seq_len: u32,
    nheads: u32,
    headdim: u32,
    d_state: u32,
    ngroups: u32,
    chunk_size: u32,
    nchunks: u32,
    d_inner: u32,
};
@group(0) @binding(8) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    // wid.x = h * nchunks * chunk_size + c * chunk_size + l
    let total_wg = params.nheads * params.nchunks * params.chunk_size;
    if (wid.x >= total_wg) { return; }

    let hcl = wid.x;
    let h = hcl / (params.nchunks * params.chunk_size);
    let cl = hcl % (params.nchunks * params.chunk_size);
    let c = cl / params.chunk_size;
    let l = cl % params.chunk_size;
    let p = lid;  // headdim index

    let t = c * params.chunk_size + l;
    if (t >= params.seq_len || p >= params.headdim) { return; }

    let heads_per_group = params.nheads / params.ngroups;
    let g = h / heads_per_group;
    let ng_ds = params.ngroups * params.d_state;

    let dA_l = dA_cumsum[(c * params.nheads + h) * params.chunk_size + l];

    // ── Intra-chunk: Σ_{s≤l} CB[l,s] * decay * dt_s * x[s,h,p] ──
    // 각 스레드가 CB를 직접 계산 (d_state=64 loop, barrier 없음)
    var intra: f32 = 0.0;
    for (var s: u32 = 0u; s <= l; s++) {
        let t_s = c * params.chunk_size + s;
        if (t_s >= params.seq_len) { break; }

        // CB dot product — 스레드당 직접 계산
        var cb: f32 = 0.0;
        for (var n: u32 = 0u; n < params.d_state; n++) {
            cb += C[t * ng_ds + g * params.d_state + n]
                * B[t_s * ng_ds + g * params.d_state + n];
        }

        let dA_s = dA_cumsum[(c * params.nheads + h) * params.chunk_size + s];
        let decay = exp(dA_l - dA_s);
        let dt_s = dt[t_s * params.nheads + h];
        intra += cb * decay * dt_s * x[t_s * params.d_inner + h * params.headdim + p];
    }

    // ── Inter-chunk ──
    let state_decay = exp(dA_l);
    var inter: f32 = 0.0;
    for (var n: u32 = 0u; n < params.d_state; n++) {
        inter += C[t * ng_ds + g * params.d_state + n]
               * prev_states[((c * params.nheads + h) * params.headdim + p) * params.d_state + n];
    }
    inter *= state_decay;

    // ── Skip + output ──
    let skip = D[h] * x[t * params.d_inner + h * params.headdim + p];
    y[t * params.d_inner + h * params.headdim + p] = intra + inter + skip;
}
