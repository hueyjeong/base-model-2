// SSD Stage 4b: Score 공유 메모리 캐싱 + output 계산
//
// score[s] = CB[l,s] * exp(dA_l - dA_s) * dt[s,h]  (p-독립)
// → 스레드 0이 score 계산, 공유 메모리에 저장 → 64 스레드가 참조
//
// dispatch: (nheads * nchunks * chunk_size, 1, 1), workgroup_size: (64, 1, 1)

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> C: array<f32>;
@group(0) @binding(2) var<storage, read> dt: array<f32>;
@group(0) @binding(3) var<storage, read> dA_cumsum: array<f32>;
@group(0) @binding(4) var<storage, read> prev_states: array<f32>;
@group(0) @binding(5) var<storage, read> D: array<f32>;
@group(0) @binding(6) var<storage, read> CB: array<f32>;
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

// score 캐싱: 최대 256개 (chunk_size). 256*4 = 1KB
var<workgroup> smem_score: array<f32, 256>;
// C[l] 캐싱: 64 floats = 256 bytes
var<workgroup> smem_c: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let total_wg = params.nheads * params.nchunks * params.chunk_size;
    if (wid.x >= total_wg) { return; }

    let hcl = wid.x;
    let h = hcl / (params.nchunks * params.chunk_size);
    let cl = hcl % (params.nchunks * params.chunk_size);
    let c = cl / params.chunk_size;
    let l = cl % params.chunk_size;
    let p = lid;

    let t = c * params.chunk_size + l;
    if (t >= params.seq_len) { return; }

    let heads_per_group = params.nheads / params.ngroups;
    let g = h / heads_per_group;
    let ng_ds = params.ngroups * params.d_state;
    let cs = params.chunk_size;
    let ch = c * params.nheads + h;

    let dA_l = dA_cumsum[ch * cs + l];

    // ── C[l] 캐싱 (64 스레드가 d_state=64를 1:1 로드) ──
    if (p < params.d_state) {
        smem_c[p] = C[t * ng_ds + g * params.d_state + p];
    }

    // ── Score 사전 계산 (64 스레드가 l+1개 score를 분담) ──
    // score[s] = CB[l,s] * exp(dA_l - dA_s) * dt[s,h]
    let effective_l = min(l + 1u, min(cs, params.seq_len - c * cs));
    for (var s = lid; s < effective_l; s += 64u) {
        let t_s = c * cs + s;
        let cb = CB[(ch * cs + l) * cs + s];
        let dA_s = dA_cumsum[ch * cs + s];
        let decay = exp(dA_l - dA_s);
        let dt_s = dt[t_s * params.nheads + h];
        smem_score[s] = cb * decay * dt_s;
    }
    workgroupBarrier();

    if (p >= params.headdim) { return; }

    // ── Intra-chunk: Σ score[s] * x[s,h,p] ──
    var intra: f32 = 0.0;
    for (var s: u32 = 0u; s < effective_l; s++) {
        let t_s = c * cs + s;
        intra += smem_score[s] * x[t_s * params.d_inner + h * params.headdim + p];
    }

    // ── Inter-chunk (smem_c 재활용) ──
    let state_decay = exp(dA_l);
    var inter: f32 = 0.0;
    for (var n: u32 = 0u; n < params.d_state; n++) {
        inter += smem_c[n]
               * prev_states[((ch * params.headdim + p) * params.d_state) + n];
    }
    inter *= state_decay;

    let skip = D[h] * x[t * params.d_inner + h * params.headdim + p];
    y[t * params.d_inner + h * params.headdim + p] = intra + inter + skip;
}
