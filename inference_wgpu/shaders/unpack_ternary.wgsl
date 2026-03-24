// Packed2Bit → packed f16 (u32에 2개씩) 언팩: 모델 로드 시 1회
// 출력: u32 배열, 각 u32 = pack2x16float(val0, val1)
// 짝수/홀수 col을 한 쌍으로 패킹

@group(0) @binding(0) var<storage, read> packed: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;

struct Params {
    rows: u32,
    cols: u32,
    packed_stride: u32,
    gamma: f32,
};
var<push_constant> params: Params;

fn decode_ternary(packed_data: array<u32, 1>, row: u32, col: u32, packed_stride: u32) -> f32 {
    let byte_idx = col / 4u;
    let bit_pos = col % 4u;
    let row_base = row * packed_stride;
    let word_idx = (row_base + byte_idx) / 4u;
    let word_off = (row_base + byte_idx) % 4u;
    // packed_data는 사용 안 함, 직접 global에서 읽음
    return 0.0; // placeholder
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // 각 스레드가 2개의 f16 값을 1개 u32로 패킹
    let pair_idx = gid.x;
    let cols_pairs = (params.cols + 1u) / 2u;
    let total_pairs = params.rows * cols_pairs;
    if (pair_idx >= total_pairs) { return; }

    let row = pair_idx / cols_pairs;
    let pair_col = pair_idx % cols_pairs;
    let col0 = pair_col * 2u;
    let col1 = col0 + 1u;

    // 첫 번째 값
    var val0: f32 = 0.0;
    if (col0 < params.cols) {
        let byte_idx = col0 / 4u;
        let bit_pos = col0 % 4u;
        let row_base = row * params.packed_stride;
        let word_idx = (row_base + byte_idx) / 4u;
        let word_off = (row_base + byte_idx) % 4u;
        let word = packed[word_idx];
        let byte_val = (word >> (word_off * 8u)) & 0xFFu;
        let shift = (3u - bit_pos) * 2u;
        let code = (byte_val >> shift) & 3u;
        if ((code & 1u) != 0u) {
            if ((code & 2u) != 0u) { val0 = -params.gamma; }
            else { val0 = params.gamma; }
        }
    }

    // 두 번째 값
    var val1: f32 = 0.0;
    if (col1 < params.cols) {
        let byte_idx = col1 / 4u;
        let bit_pos = col1 % 4u;
        let row_base = row * params.packed_stride;
        let word_idx = (row_base + byte_idx) / 4u;
        let word_off = (row_base + byte_idx) % 4u;
        let word = packed[word_idx];
        let byte_val = (word >> (word_off * 8u)) & 0xFFu;
        let shift = (3u - bit_pos) * 2u;
        let code = (byte_val >> shift) & 3u;
        if ((code & 1u) != 0u) {
            if ((code & 2u) != 0u) { val1 = -params.gamma; }
            else { val1 = params.gamma; }
        }
    }

    out[row * cols_pairs + pair_col] = pack2x16float(vec2<f32>(val0, val1));
}
