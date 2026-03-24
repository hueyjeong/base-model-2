// SSD Stage 4a: CB 행렬 사전 계산
// CB[c,h,l,s] = Σ_n C[c*cs+l, g, n] * B[c*cs+s, g, n]  (causal: s <= l)
//
// dispatch: (nchunks * nheads, 1, 1), workgroup_size: (256, 1, 1)
// 각 워크그룹 = (c,h) 쌍. 256 스레드가 (l,s) 쌍을 분담.
// chunk_size=256일 때 lower-tri = 256*257/2 = 32,896 쌍 → 스레드당 ~128 쌍

@group(0) @binding(0) var<storage, read> B: array<f32>;
@group(0) @binding(1) var<storage, read> C: array<f32>;
@group(0) @binding(2) var<storage, read_write> CB: array<f32>; // [nchunks, nheads, chunk_size, chunk_size]

struct Params {
    seq_len: u32,
    nheads: u32,
    d_state: u32,
    ngroups: u32,
    chunk_size: u32,
    nchunks: u32,
};
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let ch = wid.x;
    if (ch >= params.nchunks * params.nheads) { return; }

    let c = ch / params.nheads;
    let h = ch % params.nheads;
    let heads_per_group = params.nheads / params.ngroups;
    let g = h / heads_per_group;
    let ng_ds = params.ngroups * params.d_state;
    let cs = params.chunk_size;

    // Lower-triangular (l,s) 쌍을 256 스레드로 분담
    // 총 쌍 수 = cs*(cs+1)/2
    let n_pairs = cs * (cs + 1u) / 2u;

    for (var pair_idx = lid; pair_idx < n_pairs; pair_idx += 256u) {
        // pair_idx → (l, s) 변환: l*(l+1)/2 + s = pair_idx
        // l = floor((-1 + sqrt(1 + 8*pair_idx)) / 2)
        let l_approx = u32((-0.5 + sqrt(0.25 + 2.0 * f32(pair_idx))));
        var l = l_approx;
        // 보정: l*(l+1)/2 > pair_idx이면 l--
        if (l * (l + 1u) / 2u > pair_idx) { l -= 1u; }
        // l*(l+1)/2 + s == pair_idx에서 벗어나면 l++
        if ((l + 1u) * (l + 2u) / 2u <= pair_idx) { l += 1u; }
        let s = pair_idx - l * (l + 1u) / 2u;

        let t_l = c * cs + l;
        let t_s = c * cs + s;

        var dot: f32 = 0.0;
        if (t_l < params.seq_len && t_s < params.seq_len) {
            for (var n: u32 = 0u; n < params.d_state; n++) {
                dot += C[t_l * ng_ds + g * params.d_state + n]
                     * B[t_s * ng_ds + g * params.d_state + n];
            }
        }

        CB[(ch * cs + l) * cs + s] = dot;
    }
}
