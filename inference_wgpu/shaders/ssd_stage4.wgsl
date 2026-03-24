// SSD Stage 4: Final output (B/C 공유 메모리 캐싱 최적화)
//
// 워크그룹 = (h, c, l): 각 l에 대해 64 스레드가 p=0..63 계산
// B[s,g,:d_state]와 C[l,g,:d_state]를 공유 메모리에 캐시
// C[l]은 l 고정이므로 한 번만 로드, B[s]는 s마다 갱신
//
// dispatch: (nheads * nchunks * chunk_size, 1, 1), workgroup_size: (64, 1, 1)

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

// 공유 메모리: C[l]과 B[s] 캐시 (각 d_state=64 floats = 256 bytes)
var<workgroup> smem_c: array<f32, 64>;   // C[l, g, 0..d_state]
var<workgroup> smem_b: array<f32, 64>;   // B[s, g, 0..d_state] (매 s마다 갱신)

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

    let dA_l = dA_cumsum[(c * params.nheads + h) * params.chunk_size + l];

    // C[l] 로드 (64 스레드가 d_state=64를 1:1 로드)
    if (p < params.d_state) {
        smem_c[p] = C[t * ng_ds + g * params.d_state + p];
    }
    workgroupBarrier();

    // ── Intra-chunk ──
    var intra: f32 = 0.0;
    for (var s: u32 = 0u; s <= l; s++) {
        let t_s = c * params.chunk_size + s;
        if (t_s >= params.seq_len) { break; }

        // B[s] 공유 메모리 로드 (64 스레드가 협력)
        if (p < params.d_state) {
            smem_b[p] = B[t_s * ng_ds + g * params.d_state + p];
        }
        workgroupBarrier();

        // CB dot product — 공유 메모리에서 읽기 (전역 메모리 접근 0)
        var cb: f32 = 0.0;
        for (var n: u32 = 0u; n < params.d_state; n++) {
            cb += smem_c[n] * smem_b[n];
        }

        let dA_s = dA_cumsum[(c * params.nheads + h) * params.chunk_size + s];
        let decay = exp(dA_l - dA_s);
        let dt_s = dt[t_s * params.nheads + h];
        intra += cb * decay * dt_s * x[t_s * params.d_inner + h * params.headdim + p];

        workgroupBarrier();
    }

    if (p >= params.headdim) { return; }

    // ── Inter-chunk ──
    let state_decay = exp(dA_l);
    var inter: f32 = 0.0;
    for (var n: u32 = 0u; n < params.d_state; n++) {
        inter += smem_c[n]
               * prev_states[((c * params.nheads + h) * params.headdim + p) * params.d_state + n];
    }
    inter *= state_decay;

    let skip = D[h] * x[t * params.d_inner + h * params.headdim + p];
    y[t * params.d_inner + h * params.headdim + p] = intra + inter + skip;
}
