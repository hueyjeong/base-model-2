// Packed2Bit ternary matmul: out[t, j] = gamma * Σ_k decode(packed[j, k/4]) * x[t, k]
//
// 2-bit 인코딩: 00=0, 01=+1, 11=-1 (10 미사용)
// 리틀엔디안: u32 word의 byte[0]=LSB, byte 내 bit[7:6]=첫 번째 값
//
// Ternary 모드 (mode=0): out = gamma * acc
// BitLinear 모드 (mode=1): out = gamma * token_scales[col] * acc

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
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> token_scales: array<f32>;
@group(0) @binding(5) var<storage, read> row_sums: array<i32>;

// 타일 설정: 각 스레드가 4×4 출력 계산 → 워크그룹당 32×32 출력
const WG: u32 = 8u;       // workgroup_size(8, 8)
const BM: u32 = 32u;      // 출력 tile M 크기 (WG * TM)
const BN: u32 = 32u;      // 출력 tile N 크기 (WG * TN)
const BK: u32 = 32u;      // K-strip 크기
const TM: u32 = 4u;       // 스레드당 M 출력
const TN: u32 = 4u;       // 스레드당 N 출력

// 공유 메모리: BN×BK + BM×BK = 32*32 + 32*32 = 2048 + 2048 = 4096 floats = 16KB
var<workgroup> smem_x: array<f32, 1024>;  // BN × BK = 32 × 32
var<workgroup> smem_w: array<f32, 1024>;  // BM × BK = 32 × 32

fn decode_ternary_fast(packed_byte: u32, pos: u32) -> f32 {
    let shift = (3u - pos) * 2u;
    let code = (packed_byte >> shift) & 3u;
    // 분기 없는 디코딩: code→f32 LUT
    // 00=0, 01=+1, 10=0, 11=-1
    // (code & 1) gives 0 for 0/2, 1 for 1/3
    // (code >> 1) gives 0 for 0/1, 1 for 2/3
    // sign = 1 - 2*(code>>1) = +1 or -1
    // mask = code & 1  (0 or 1)
    // result = mask * sign = 0, +1, 0, -1
    let mask = f32(code & 1u);
    let sign = 1.0 - 2.0 * f32(code >> 1u);
    return mask * sign;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {

    let tx = lid.x;  // 0..7
    let ty = lid.y;  // 0..7
    let bx = wid.x;  // block col (N 방향)
    let by = wid.y;  // block row (M 방향)

    // 이 스레드의 4×4 누적기
    var acc: array<f32, 16>;  // TM × TN = 4×4
    for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }

    let n_base = bx * BN;  // 이 블록의 N 시작
    let m_base = by * BM;  // 이 블록의 M 시작
    let tid = ty * WG + tx; // 0..63

    for (var k_start: u32 = 0u; k_start < params.K; k_start += BK) {
        // ── 공유 메모리 로드: smem_x[BN, BK] ──
        // 64 스레드, 32×32=1024 원소 → 스레드당 16개
        for (var i = 0u; i < 16u; i++) {
            let flat = tid * 16u + i;
            let sn = flat / BK;  // 0..31 (N 차원)
            let sk = flat % BK;  // 0..31 (K 차원)
            let gn = n_base + sn;
            let gk = k_start + sk;
            if (gn < params.N && gk < params.K) {
                smem_x[sn * BK + sk] = x[gn * params.K + gk];
            } else {
                smem_x[sn * BK + sk] = 0.0;
            }
        }

        // ── 공유 메모리 로드: smem_w[BM, BK] ── (packed2bit 디코드)
        for (var i = 0u; i < 16u; i++) {
            let flat = tid * 16u + i;
            let sm = flat / BK;  // 0..31 (M 차원)
            let sk = flat % BK;  // 0..31 (K 차원)
            let gm = m_base + sm;
            let gk = k_start + sk;
            if (gm < params.M && gk < params.K) {
                let byte_idx = gk / 4u;
                let byte_pos = gk % 4u;
                let row_base = gm * params.packed_stride;
                let word_idx = (row_base + byte_idx) / 4u;
                let word_off = (row_base + byte_idx) % 4u;
                let word = packed_w[word_idx];
                let pbyte = (word >> (word_off * 8u)) & 0xFFu;
                smem_w[sm * BK + sk] = decode_ternary_fast(pbyte, byte_pos);
            } else {
                smem_w[sm * BK + sk] = 0.0;
            }
        }

        workgroupBarrier();

        // ── 계산: 각 스레드가 TM×TN 출력 누적 ──
        for (var k = 0u; k < BK; k++) {
            // 이 스레드의 w 값 TM개 로드
            var w_reg: array<f32, 4>;
            for (var tm = 0u; tm < TM; tm++) {
                w_reg[tm] = smem_w[(ty * TM + tm) * BK + k];
            }
            // 이 스레드의 x 값 TN개 로드
            var x_reg: array<f32, 4>;
            for (var tn = 0u; tn < TN; tn++) {
                x_reg[tn] = smem_x[(tx * TN + tn) * BK + k];
            }
            // outer product 누적
            for (var tm = 0u; tm < TM; tm++) {
                for (var tn = 0u; tn < TN; tn++) {
                    acc[tm * TN + tn] += w_reg[tm] * x_reg[tn];
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
