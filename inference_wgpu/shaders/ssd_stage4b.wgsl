// SSD Stage 4b: Score 계산 + Intra-chunk output + Inter-chunk + Skip
//
// score[s] = CB[l,s] * exp(dA_l - dA_s) * dt[s,h]  (p-독립, 한 번만 계산)
// intra[p] = Σ_{s=0..l} score[s] * x[s, h, p]
// inter[p] = exp(dA_l) * Σ_n C[l,g,n] * prev_states[c,h,p,n]
// y[t, h*hd+p] = intra[p] + inter[p] + D[h] * x[t, h*hd+p]
//
// dispatch: (nheads * nchunks * chunk_size, 1, 1), workgroup_size: (64, 1, 1)
// 각 워크그룹 = (h,c,l), 64 스레드 = p 차원

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> C: array<f32>;
@group(0) @binding(2) var<storage, read> dt: array<f32>;
@group(0) @binding(3) var<storage, read> dA_cumsum: array<f32>;
@group(0) @binding(4) var<storage, read> prev_states: array<f32>;
@group(0) @binding(5) var<storage, read> D: array<f32>;
@group(0) @binding(6) var<storage, read> CB: array<f32>;  // [nchunks, nheads, cs, cs]
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
    let total_wg = params.nheads * params.nchunks * params.chunk_size;
    if (wid.x >= total_wg) { return; }

    let hcl = wid.x;
    let h = hcl / (params.nchunks * params.chunk_size);
    let cl = hcl % (params.nchunks * params.chunk_size);
    let c = cl / params.chunk_size;
    let l = cl % params.chunk_size;
    let p = lid;

    let t = c * params.chunk_size + l;
    if (t >= params.seq_len || p >= params.headdim) { return; }

    let heads_per_group = params.nheads / params.ngroups;
    let g = h / heads_per_group;
    let ng_ds = params.ngroups * params.d_state;
    let cs = params.chunk_size;
    let ch = c * params.nheads + h;

    let dA_l = dA_cumsum[ch * cs + l];

    // ── Intra-chunk: score[s] * x[s,h,p] 누적 ──
    // score는 p-독립이지만 공유 메모리 없이 각 스레드가 직접 계산 (CB 이미 사전 계산됨)
    var intra: f32 = 0.0;
    for (var s: u32 = 0u; s <= l; s++) {
        let t_s = c * cs + s;
        if (t_s >= params.seq_len) { break; }

        let cb = CB[(ch * cs + l) * cs + s];
        let dA_s = dA_cumsum[ch * cs + s];
        let decay = exp(dA_l - dA_s);
        let dt_s = dt[t_s * params.nheads + h];
        let score = cb * decay * dt_s;

        intra += score * x[t_s * params.d_inner + h * params.headdim + p];
    }

    // ── Inter-chunk ──
    let state_decay = exp(dA_l);
    var inter: f32 = 0.0;
    for (var n: u32 = 0u; n < params.d_state; n++) {
        inter += C[t * ng_ds + g * params.d_state + n]
               * prev_states[((ch * params.headdim + p) * params.d_state) + n];
    }
    inter *= state_decay;

    // ── Skip + output ──
    let skip = D[h] * x[t * params.d_inner + h * params.headdim + p];
    y[t * params.d_inner + h * params.headdim + p] = intra + inter + skip;
}
