// SSD Stage 1: dA cumulative sum (병렬 Hillis-Steele scan)
// 각 (chunk, head) 쌍에 대해 inclusive prefix sum
// dA_cumsum[c,h,l] = Σ_{i=0..l} A[h] * dt[c*chunk_size+i, h]

@group(0) @binding(0) var<storage, read> dt: array<f32>;      // [seq_len, nheads]
@group(0) @binding(1) var<storage, read> a_neg: array<f32>;    // [nheads]
@group(0) @binding(2) var<storage, read_write> dA_cumsum: array<f32>; // [nchunks, nheads, chunk_size]

struct Params {
    seq_len: u32,
    nheads: u32,
    chunk_size: u32,
    nchunks: u32,
};
var<push_constant> params: Params;

var<workgroup> smem: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let ch = wid.x;
    if (ch >= params.nchunks * params.nheads) { return; }

    let c = ch / params.nheads;
    let h = ch % params.nheads;
    let l = lid;
    let t = c * params.chunk_size + l;

    // 각 스레드가 자신의 dA값 로드
    var val: f32 = 0.0;
    if (l < params.chunk_size && t < params.seq_len) {
        val = a_neg[h] * dt[t * params.nheads + h];
    }
    smem[l] = val;
    workgroupBarrier();

    // Hillis-Steele inclusive prefix sum
    // O(n log n) work, O(log n) steps
    for (var stride = 1u; stride < params.chunk_size; stride *= 2u) {
        var tmp: f32 = 0.0;
        if (l >= stride && l < params.chunk_size) {
            tmp = smem[l - stride];
        }
        workgroupBarrier();
        if (l < params.chunk_size) {
            smem[l] += tmp;
        }
        workgroupBarrier();
    }

    // 결과 기록
    if (l < params.chunk_size) {
        dA_cumsum[ch * params.chunk_size + l] = smem[l];
    }
}
