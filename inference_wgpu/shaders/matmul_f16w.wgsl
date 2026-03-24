// Packed F16 weight × F32 activation matmul
// weight: u32[M, K/2] (pack2x16float), x: f32[N, K], out: f32[N, M]
// gamma가 이미 weight에 적용됨
//
// 64×64 output tile, 16×16 workgroup (256 threads), 4×4 per thread
// 가중치를 f16으로 저장하여 메모리 대역폭 절반
//
// mode=0: out = W @ X^T
// mode=1: out = (W @ X^T) * token_scales[t]

@group(0) @binding(0) var<storage, read> weight_packed: array<u32>;  // packed f16 pairs
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> token_scales: array<f32>;

struct Params {
    M: u32,
    N: u32,
    K: u32,
    mode: u32,
};
var<push_constant> params: Params;

const WG: u32 = 16u;
const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 32u;
const TM: u32 = 4u;
const TN: u32 = 4u;

var<workgroup> smem_x: array<f32, 2048>;  // BN × BK = 64 × 32 f32 = 8KB
var<workgroup> smem_w: array<f32, 2048>;  // BM × BK = 64 × 32 f32 = 8KB (언팩 후)

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let n_base = wid.x * BN;
    let m_base = wid.y * BM;
    let tid = ty * WG + tx;

    let k_pairs = (params.K + 1u) / 2u;  // K/2 (packed pairs per row)

    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }

    for (var k_start: u32 = 0u; k_start < params.K; k_start += BK) {
        // x tile 로드 (f32): 256 threads × 8 = 2048
        for (var i = 0u; i < 8u; i++) {
            let flat = tid * 8u + i;
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

        // w tile 로드 (packed f16 → f32 언팩): 256 threads × 8 = 2048
        for (var i = 0u; i < 8u; i++) {
            let flat = tid * 8u + i;
            let sm = flat / BK;
            let sk = flat % BK;
            let gm = m_base + sm;
            let gk = k_start + sk;
            if (gm < params.M && gk < params.K) {
                // packed f16 pair에서 해당 값 추출
                let pair_idx = gk / 2u;
                let pair_off = gk % 2u;
                let packed_val = weight_packed[gm * k_pairs + pair_idx];
                let unpacked = unpack2x16float(packed_val);
                smem_w[sm * BK + sk] = unpacked[pair_off];
            } else {
                smem_w[sm * BK + sk] = 0.0;
            }
        }

        workgroupBarrier();

        for (var k = 0u; k < BK; k++) {
            var w_reg: array<f32, 4>;
            for (var tm = 0u; tm < TM; tm++) {
                w_reg[tm] = smem_w[(ty * TM + tm) * BK + k];
            }
            var x_reg: array<f32, 4>;
            for (var tn = 0u; tn < TN; tn++) {
                x_reg[tn] = smem_x[(tx * TN + tn) * BK + k];
            }
            for (var tm = 0u; tm < TM; tm++) {
                for (var tn = 0u; tn < TN; tn++) {
                    acc[tm * TN + tn] += w_reg[tm] * x_reg[tn];
                }
            }
        }

        workgroupBarrier();
    }

    for (var tm = 0u; tm < TM; tm++) {
        let gm = m_base + ty * TM + tm;
        if (gm >= params.M) { continue; }
        for (var tn = 0u; tn < TN; tn++) {
            let gn = n_base + tx * TN + tn;
            if (gn >= params.N) { continue; }
            var val = acc[tm * TN + tn];
            if (params.mode == 1u) {
                val *= token_scales[gn];
            }
            out[gn * params.M + gm] = val;
        }
    }
}
