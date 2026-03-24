// Packed2Bit ternary matmul: 곱셈 없는 조건부 add/sub
//
// out[t, j] = gamma * Σ_k ternary(w[j,k]) * x[t, k]
// ternary: {-1, 0, +1} → multiply 대신 add/sub/skip
//
// 32×32 output tile, 8×8 workgroup, 스레드당 4×4 출력
// 공유 메모리: x는 f32, w는 packed u32 (4 ternary per byte)

@group(0) @binding(0) var<storage, read> packed_w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Params {
    M: u32,
    N: u32,
    K: u32,
    gamma: f32,
    packed_stride: u32,
    mode: u32,
};
var<push_constant> params: Params;
@group(0) @binding(3) var<storage, read> token_scales: array<f32>;
@group(0) @binding(4) var<storage, read> row_sums: array<i32>;

const WG: u32 = 8u;
const BM: u32 = 32u;
const BN: u32 = 32u;
const BK: u32 = 32u;   // K-strip, 4의 배수 (packed byte 정렬)
const TM: u32 = 4u;
const TN: u32 = 4u;

// 공유 메모리: x는 f32, w는 packed u32 (4 ternary per u32 byte)
var<workgroup> smem_x: array<f32, 1024>;          // BN × BK = 32 × 32
var<workgroup> smem_w_packed: array<u32, 256>;     // BM × (BK/4) = 32 × 8 packed bytes as u32

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * WG + tx;
    let n_base = wid.x * BN;
    let m_base = wid.y * BM;

    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }

    for (var k_start: u32 = 0u; k_start < params.K; k_start += BK) {
        // ── x tile 로드 (f32) ──
        for (var i = 0u; i < 16u; i++) {
            let flat = tid * 16u + i;
            let sn = flat / BK;
            let sk = flat % BK;
            let gn = n_base + sn;
            let gk = k_start + sk;
            if (gn < params.N && gk < params.K) {
                smem_x[sn * BK + sk] = x[gn * params.K + gk];
            } else {
                smem_x[sn * BK + sk] = 0.0;
            }
        }

        // ── w tile 로드 (packed bytes → u32) ──
        let packed_per_row = BK / 4u;  // 8
        for (var i = 0u; i < 4u; i++) {
            let flat = tid * 4u + i;
            let sm = flat / packed_per_row;
            let sb = flat % packed_per_row;
            let gm = m_base + sm;
            let gk_byte = (k_start / 4u) + sb;
            if (gm < params.M && gk_byte * 4u < params.K) {
                let row_base = gm * params.packed_stride;
                let word_idx = (row_base + gk_byte) / 4u;
                let word_off = (row_base + gk_byte) % 4u;
                let word = packed_w[word_idx];
                smem_w_packed[sm * packed_per_row + sb] = (word >> (word_off * 8u)) & 0xFFu;
            } else {
                smem_w_packed[sm * packed_per_row + sb] = 0u;
            }
        }

        workgroupBarrier();

        // ── 계산: conditional add/sub (원래 방식, 분기가 GPU에서 더 빠름) ──
        for (var kb = 0u; kb < BK / 4u; kb++) {
            var x_vals: array<array<f32, 4>, 4>;
            for (var tn = 0u; tn < TN; tn++) {
                let x_base = (tx * TN + tn) * BK + kb * 4u;
                x_vals[tn][0] = smem_x[x_base];
                x_vals[tn][1] = smem_x[x_base + 1u];
                x_vals[tn][2] = smem_x[x_base + 2u];
                x_vals[tn][3] = smem_x[x_base + 3u];
            }

            for (var tm = 0u; tm < TM; tm++) {
                let packed_byte = smem_w_packed[(ty * TM + tm) * (BK / 4u) + kb];
                let c0 = (packed_byte >> 6u) & 3u;
                let c1 = (packed_byte >> 4u) & 3u;
                let c2 = (packed_byte >> 2u) & 3u;
                let c3 = packed_byte & 3u;

                for (var tn = 0u; tn < TN; tn++) {
                    let idx = tm * TN + tn;
                    if ((c0 & 1u) != 0u) {
                        if ((c0 & 2u) != 0u) { acc[idx] -= x_vals[tn][0]; }
                        else                  { acc[idx] += x_vals[tn][0]; }
                    }
                    if ((c1 & 1u) != 0u) {
                        if ((c1 & 2u) != 0u) { acc[idx] -= x_vals[tn][1]; }
                        else                  { acc[idx] += x_vals[tn][1]; }
                    }
                    if ((c2 & 1u) != 0u) {
                        if ((c2 & 2u) != 0u) { acc[idx] -= x_vals[tn][2]; }
                        else                  { acc[idx] += x_vals[tn][2]; }
                    }
                    if ((c3 & 1u) != 0u) {
                        if ((c3 & 2u) != 0u) { acc[idx] -= x_vals[tn][3]; }
                        else                  { acc[idx] += x_vals[tn][3]; }
                    }
                }
            }
        }

        workgroupBarrier();
    }

    // ── 결과 저장 ──
    for (var tm = 0u; tm < TM; tm++) {
        let gm = m_base + ty * TM + tm;
        if (gm >= params.M) { continue; }
        for (var tn = 0u; tn < TN; tn++) {
            let gn = n_base + tx * TN + tn;
            if (gn >= params.N) { continue; }
            var val = acc[tm * TN + tn];
            if (params.mode == 1u) {
                val *= token_scales[gn] * params.gamma;
            } else {
                val *= params.gamma;
            }
            out[gn * params.M + gm] = val;
        }
    }
}
