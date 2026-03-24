// SSD Stage 4: Final output computation (CB 공유 메모리 최적화)
// y[t, h*headdim+p] = intra_chunk + inter_chunk + D[h] * x[t, h*headdim+p]
//
// CB[l,s] = dot(C[l,:], B[s,:]) — 워크그룹 내 64 스레드로 병렬 reduction
// 그 후 각 스레드가 자신의 p 차원 출력을 계산
//
// dispatch: (nheads * nchunks, 1, 1), workgroup_size: (64, 1, 1)

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

// CB dot product 병렬 reduction용 공유 메모리
var<workgroup> shared_cb: f32;
var<workgroup> shared_partial: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let wg = gid.x / 64u;
    if (wg >= params.nheads * params.nchunks) { return; }

    let h = wg / params.nchunks;
    let c = wg % params.nchunks;
    let p = lid;  // headdim index (0..63)
    if (p >= params.headdim) { return; }

    let heads_per_group = params.nheads / params.ngroups;
    let g = h / heads_per_group;
    let ng_ds = params.ngroups * params.d_state;

    for (var l: u32 = 0u; l < params.chunk_size; l++) {
        let t = c * params.chunk_size + l;
        if (t >= params.seq_len) { break; }

        let dA_l = dA_cumsum[(c * params.nheads + h) * params.chunk_size + l];

        // Intra-chunk: Σ_{s≤l} CB[l,s] * decay * dt_s * x[s,h,p]
        var intra: f32 = 0.0;
        for (var s: u32 = 0u; s <= l; s++) {
            let t_s = c * params.chunk_size + s;
            if (t_s >= params.seq_len) { break; }

            // CB[l,s] 병렬 reduction — 64 스레드가 d_state=64를 분할
            // 각 스레드가 1개의 C[l,n]*B[s,n] 계산 (d_state==headdim==64)
            var my_cb: f32 = 0.0;
            if (p < params.d_state) {
                my_cb = C[t * ng_ds + g * params.d_state + p]
                      * B[t_s * ng_ds + g * params.d_state + p];
            }
            shared_partial[p] = my_cb;
            workgroupBarrier();

            // Tree reduction (64 → 1)
            if (p < 32u) { shared_partial[p] += shared_partial[p + 32u]; }
            workgroupBarrier();
            if (p < 16u) { shared_partial[p] += shared_partial[p + 16u]; }
            workgroupBarrier();
            if (p < 8u) { shared_partial[p] += shared_partial[p + 8u]; }
            workgroupBarrier();
            if (p < 4u) { shared_partial[p] += shared_partial[p + 4u]; }
            workgroupBarrier();
            if (p < 2u) { shared_partial[p] += shared_partial[p + 2u]; }
            workgroupBarrier();
            if (p == 0u) { shared_cb = shared_partial[0] + shared_partial[1]; }
            workgroupBarrier();

            let cb = shared_cb;
            let dA_s = dA_cumsum[(c * params.nheads + h) * params.chunk_size + s];
            let decay = exp(dA_l - dA_s);
            let dt_s = dt[t_s * params.nheads + h];
            intra += cb * decay * dt_s * x[t_s * params.d_inner + h * params.headdim + p];
        }

        // Inter-chunk: Σ_n C[t,g,n] * prev_states[c,h,p,n] * decay
        let state_decay = exp(dA_l);
        var inter: f32 = 0.0;
        for (var n: u32 = 0u; n < params.d_state; n++) {
            inter += C[t * ng_ds + g * params.d_state + n]
                   * prev_states[((c * params.nheads + h) * params.headdim + p) * params.d_state + n];
        }
        inter *= state_decay;

        let skip = D[h] * x[t * params.d_inner + h * params.headdim + p];
        y[t * params.d_inner + h * params.headdim + p] = intra + inter + skip;
    }
}
