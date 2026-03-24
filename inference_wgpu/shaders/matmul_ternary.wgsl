// Packed2Bit ternary matmul: out[t, j] = gamma * Σ_k decode(packed[j, k/4]) * x[t, k]
//
// 2-bit 인코딩: 00=0, 01=+1, 11=-1 (10 미사용)
// packed_w: u32 배열로 읽음 (wgsl에 u8 array 없음)
// weight: [M rows, packed_stride bytes/row] → [M, ceil(K/4)] packed bytes
//
// BitLinear 모드 (mode=1): x는 이미 LayerNorm + quantize 완료된 값
//   out에 per-token scale * gamma 적용
//
// Ternary 모드 (mode=0): out에 gamma만 적용

@group(0) @binding(0) var<storage, read> packed_w: array<u32>;  // packed 2-bit weights
@group(0) @binding(1) var<storage, read> x: array<f32>;         // [N, K]
@group(0) @binding(2) var<storage, read_write> out: array<f32>;  // [N, M]

struct Params {
    M: u32,             // weight rows (output dim)
    N: u32,             // batch (seq_len)
    K: u32,             // input dim (weight cols)
    gamma: f32,         // ternary scale
    packed_stride: u32, // bytes per packed row
    mode: u32,          // 0=ternary, 1=bitlinear
};
@group(0) @binding(3) var<uniform> params: Params;

// BitLinear 모드 전용: per-token scale
@group(0) @binding(4) var<storage, read> token_scales: array<f32>;  // [N]
// BitLinear 모드 전용: row_sums (bias correction)
@group(0) @binding(5) var<storage, read> row_sums: array<i32>;  // [M]

const TILE: u32 = 16u;
const STRIP: u32 = 64u;

var<workgroup> tile_x: array<f32, 1024>;  // TILE × STRIP
var<workgroup> tile_w: array<f32, 1024>;  // TILE × STRIP

// packed byte에서 특정 위치의 ternary 값 디코드
fn decode_ternary(packed_byte: u32, pos: u32) -> f32 {
    // pos: 바이트 내 위치 (0=MSB쌍, 3=LSB쌍)
    let shift = (3u - pos) * 2u;
    let code = (packed_byte >> shift) & 3u;
    // 00=0, 01=+1, 11=-1
    if (code == 1u) { return 1.0; }
    if (code == 3u) { return -1.0; }
    return 0.0;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let col = gid.x;  // N 차원 (batch token)
    let row = gid.y;  // M 차원 (output dim)

    var acc: f32 = 0.0;

    for (var k_start: u32 = 0u; k_start < params.K; k_start += STRIP) {
        // x tile 로드
        for (var i: u32 = 0u; i < 4u; i++) {
            let k = lid.y * 4u + i;
            let k_abs = k_start + k;
            if (col < params.N && k_abs < params.K) {
                tile_x[lid.x * STRIP + k] = x[col * params.K + k_abs];
            } else {
                tile_x[lid.x * STRIP + k] = 0.0;
            }
        }

        // weight tile 로드 + 2-bit 언팩
        for (var i: u32 = 0u; i < 4u; i++) {
            let k = lid.x * 4u + i;
            let k_abs = k_start + k;
            if (row < params.M && k_abs < params.K) {
                // packed byte 위치 계산
                let byte_idx = k_abs / 4u;
                let byte_pos = k_abs % 4u;
                // u32 배열에서 byte 추출
                let row_base = row * params.packed_stride;
                let word_idx = (row_base + byte_idx) / 4u;
                let word_off = (row_base + byte_idx) % 4u;
                let word = packed_w[word_idx];
                let pbyte = (word >> (word_off * 8u)) & 0xFFu;
                tile_w[lid.y * STRIP + k] = decode_ternary(pbyte, byte_pos);
            } else {
                tile_w[lid.y * STRIP + k] = 0.0;
            }
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < STRIP; k++) {
            acc += tile_w[lid.y * STRIP + k] * tile_x[lid.x * STRIP + k];
        }

        workgroupBarrier();
    }

    if (row < params.M && col < params.N) {
        if (params.mode == 1u) {
            // BitLinear: y = (acc - row_sums[row] * 128) * token_scales[col] * gamma
            // i8_sgemm 호환: u8 offset correction
            let corrected = acc;  // x는 이미 float이므로 correction 불필요
            out[col * params.M + row] = corrected * token_scales[col] * params.gamma;
        } else {
            out[col * params.M + row] = acc * params.gamma;
        }
    }
}
