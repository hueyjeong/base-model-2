// F32 tiled matmul: out[t, j] = Σ_k weight[j, k] * x[t, k]
// weight: [M, K] row-major, x: [N, K] row-major, out: [N, M] row-major
// 16×16 output tile, K-strip 64

@group(0) @binding(0) var<storage, read> weight: array<f32>;  // [M, K]
@group(0) @binding(1) var<storage, read> x: array<f32>;       // [N, K]
@group(0) @binding(2) var<storage, read_write> out: array<f32>; // [N, M]

struct Params {
    M: u32,  // weight rows (output dim)
    N: u32,  // batch (seq_len)
    K: u32,  // input dim
};
@group(0) @binding(3) var<uniform> params: Params;

const TILE: u32 = 16u;
const STRIP: u32 = 64u;

var<workgroup> tile_x: array<f32, 1024>;   // TILE × STRIP = 16 × 64
var<workgroup> tile_w: array<f32, 1024>;   // TILE × STRIP = 16 × 64

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let col = gid.x;  // N 차원 (batch/seq token)
    let row = gid.y;  // M 차원 (output dim)

    var acc: f32 = 0.0;

    for (var k_start: u32 = 0u; k_start < params.K; k_start += STRIP) {
        // 공유 메모리로 x tile 로드: tile_x[lid.x][k] = x[col, k_start+k]
        for (var i: u32 = 0u; i < 4u; i++) {
            let k = lid.y * 4u + i;
            let k_abs = k_start + k;
            if (col < params.N && k_abs < params.K) {
                tile_x[lid.x * STRIP + k] = x[col * params.K + k_abs];
            } else {
                tile_x[lid.x * STRIP + k] = 0.0;
            }
        }

        // 공유 메모리로 weight tile 로드: tile_w[lid.y][k] = weight[row, k_start+k]
        for (var i: u32 = 0u; i < 4u; i++) {
            let k = lid.x * 4u + i;
            let k_abs = k_start + k;
            if (row < params.M && k_abs < params.K) {
                tile_w[lid.y * STRIP + k] = weight[row * params.K + k_abs];
            } else {
                tile_w[lid.y * STRIP + k] = 0.0;
            }
        }

        workgroupBarrier();

        // dot product
        for (var k: u32 = 0u; k < STRIP; k++) {
            acc += tile_w[lid.y * STRIP + k] * tile_x[lid.x * STRIP + k];
        }

        workgroupBarrier();
    }

    if (row < params.M && col < params.N) {
        out[col * params.M + row] = acc;
    }
}
