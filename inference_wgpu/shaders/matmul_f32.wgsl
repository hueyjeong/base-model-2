// F32 tiled matmul: out[t, j] = Σ_k weight[j, k] * x[t, k]
// weight: [M, K] row-major, x: [N, K] row-major, out: [N, M] row-major
// 32×32 output tile, 스레드당 4×4 출력

@group(0) @binding(0) var<storage, read> weight: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Params {
    M: u32,
    N: u32,
    K: u32,
};
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 8u;
const BM: u32 = 32u;
const BN: u32 = 32u;
const BK: u32 = 32u;
const TM: u32 = 4u;
const TN: u32 = 4u;

var<workgroup> smem_x: array<f32, 1024>;  // BN × BK
var<workgroup> smem_w: array<f32, 1024>;  // BM × BK

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let n_base = wid.x * BN;
    let m_base = wid.y * BM;
    let tid = ty * WG + tx;

    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }

    for (var k_start: u32 = 0u; k_start < params.K; k_start += BK) {
        // 공유 메모리 로드 (64 스레드, 1024 원소 → 스레드당 16개)
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
        for (var i = 0u; i < 16u; i++) {
            let flat = tid * 16u + i;
            let sm = flat / BK;
            let sk = flat % BK;
            let gm = m_base + sm;
            let gk = k_start + sk;
            if (gm < params.M && gk < params.K) {
                smem_w[sm * BK + sk] = weight[gm * params.K + gk];
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
            out[gn * params.M + gm] = acc[tm * TN + tn];
        }
    }
}
