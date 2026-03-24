// SSD Stage 4a: CB 행렬 = C @ B^T (tiled matmul, causal mask)
//
// CB[c,h,l,s] = Σ_n C[c*cs+l, g, n] * B[c*cs+s, g, n]  (s <= l만 유효)
// C: [seq, ng*ds], B: [seq, ng*ds]
// CB: [nchunks*nheads, chunk_size, chunk_size]
//
// Tiled matmul: chunk_size=256, d_state=64
// 각 (c,h) 쌍에 대해 [256, 64] @ [256, 64]^T = [256, 256]
// 워크그룹: 8×8, 스레드당 4×4 출력 → 32×32 tile
// dispatch: (ceil(cs/32) * ceil(cs/32), nchunks * nheads, 1)

@group(0) @binding(0) var<storage, read> B: array<f32>;
@group(0) @binding(1) var<storage, read> C: array<f32>;
@group(0) @binding(2) var<storage, read_write> CB: array<f32>;

struct Params {
    seq_len: u32,
    nheads: u32,
    d_state: u32,
    ngroups: u32,
    chunk_size: u32,
    nchunks: u32,
};
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 8u;
const BM: u32 = 32u;   // tile L 크기
const BN: u32 = 32u;   // tile S 크기
const BK: u32 = 32u;   // d_state strip (64면 2회)
const TM: u32 = 4u;
const TN: u32 = 4u;

var<workgroup> smem_c: array<f32, 1024>;  // BM × BK = 32 × 32
var<workgroup> smem_b: array<f32, 1024>;  // BN × BK = 32 × 32

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tx = lid.x;  // S 방향 (0..7)
    let ty = lid.y;  // L 방향 (0..7)
    let tid = ty * WG + tx;

    // wid.x = tile index within chunk (L/S tiles)
    // wid.y = c * nheads + h
    let ch = wid.y;
    if (ch >= params.nchunks * params.nheads) { return; }

    let c = ch / params.nheads;
    let h = ch % params.nheads;
    let heads_per_group = params.nheads / params.ngroups;
    let g = h / heads_per_group;
    let ng_ds = params.ngroups * params.d_state;
    let cs = params.chunk_size;

    // tile 위치
    let tiles_per_row = (cs + BN - 1u) / BN;
    let tile_l = wid.x / tiles_per_row;  // L 방향 tile
    let tile_s = wid.x % tiles_per_row;  // S 방향 tile

    let l_base = tile_l * BM;
    let s_base = tile_s * BN;

    // 누적기
    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }

    for (var k_start: u32 = 0u; k_start < params.d_state; k_start += BK) {
        // smem_c: C[c*cs+l_base+ty*4+tm, g, k_start+k]
        for (var i = 0u; i < 16u; i++) {
            let flat = tid * 16u + i;
            let sl = flat / BK;
            let sk = flat % BK;
            let gl = l_base + sl;
            let gk = k_start + sk;
            let t = c * cs + gl;
            if (gl < cs && t < params.seq_len && gk < params.d_state) {
                smem_c[sl * BK + sk] = C[t * ng_ds + g * params.d_state + gk];
            } else {
                smem_c[sl * BK + sk] = 0.0;
            }
        }
        // smem_b: B[c*cs+s_base+..., g, k_start+k]
        for (var i = 0u; i < 16u; i++) {
            let flat = tid * 16u + i;
            let ss = flat / BK;
            let sk = flat % BK;
            let gs = s_base + ss;
            let gk = k_start + sk;
            let t = c * cs + gs;
            if (gs < cs && t < params.seq_len && gk < params.d_state) {
                smem_b[ss * BK + sk] = B[t * ng_ds + g * params.d_state + gk];
            } else {
                smem_b[ss * BK + sk] = 0.0;
            }
        }

        workgroupBarrier();

        for (var k = 0u; k < BK; k++) {
            var c_reg: array<f32, 4>;
            for (var tm = 0u; tm < TM; tm++) {
                c_reg[tm] = smem_c[(ty * TM + tm) * BK + k];
            }
            var b_reg: array<f32, 4>;
            for (var tn = 0u; tn < TN; tn++) {
                b_reg[tn] = smem_b[(tx * TN + tn) * BK + k];
            }
            for (var tm = 0u; tm < TM; tm++) {
                for (var tn = 0u; tn < TN; tn++) {
                    acc[tm * TN + tn] += c_reg[tm] * b_reg[tn];
                }
            }
        }

        workgroupBarrier();
    }

    // 저장 (causal mask: s > l인 부분은 0)
    let cb_base = ch * cs * cs;
    for (var tm = 0u; tm < TM; tm++) {
        let gl = l_base + ty * TM + tm;
        if (gl >= cs) { continue; }
        for (var tn = 0u; tn < TN; tn++) {
            let gs = s_base + tx * TN + tn;
            if (gs >= cs || gs > gl) { continue; }
            let t_l = c * cs + gl;
            let t_s = c * cs + gs;
            if (t_l < params.seq_len && t_s < params.seq_len) {
                CB[cb_base + gl * cs + gs] = acc[tm * TN + tn];
            }
        }
    }
}
