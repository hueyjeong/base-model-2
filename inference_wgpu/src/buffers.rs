//! GPU 버퍼 관리: 가중치 버퍼 + 활성화 풀
//!
//! 가중치: 모델 로드 시 1회 생성 (불변)
//! 활성화: 레이어 간 핑퐁 재사용 (가변)

use wgpu::*;
use wgpu::util::DeviceExt;

use crate::gpu::GpuContext;

/// 활성화 버퍼 풀 — 레이어 간 재사용
pub struct ActivationPool {
    /// seq × max(6144, d_in_proj) — matmul 출력, proj 결과
    pub buf_a: Buffer,
    /// seq × max(3072, d_in_proj) — FFN 중간, mamba2 중간
    pub buf_b: Buffer,
    /// seq × d_model (hidden state 핑퐁 1)
    pub buf_c: Buffer,
    /// seq × d_model (hidden state 핑퐁 2)
    pub buf_d: Buffer,
    /// seq × max(d_inner, d_ff) — 추가 임시 (버퍼 충돌 방지)
    pub buf_e: Buffer,
    /// seq × d_model — mixing 출력 임시
    pub buf_f: Buffer,
    /// Mamba2: xbc [seq × d_conv_in]
    pub xbc: Buffer,
    /// Mamba2: xbc_conv [seq × d_conv_in] (conv1d 출력)
    pub xbc_conv: Buffer,
    /// Mamba2: x_conv [seq × d_inner]
    pub x_conv: Buffer,
    /// Mamba2: b_conv [seq × ngroups*d_state]
    pub b_conv: Buffer,
    /// Mamba2: c_conv [seq × ngroups*d_state]
    pub c_conv: Buffer,
    /// Mamba2: dt [seq × nheads]
    pub dt: Buffer,
    /// SSD: CB scores [nchunks × nheads × chunk_size × chunk_size]
    pub ssd_cb: Buffer,
    /// SSD: dA_cumsum [nchunks × nheads × chunk_size]
    pub ssd_da_cumsum: Buffer,
    /// SSD: chunk_states [nchunks × nheads × headdim × d_state]
    pub ssd_chunk_states: Buffer,
    /// SSD: prev_states (동일 크기)
    pub ssd_prev_states: Buffer,
    /// per-token scalar 임시 (norm reduction 등) [seq_len]
    pub scalars: Buffer,
    /// 결과 tags [seq_len] u32
    pub tags: Buffer,
    /// Staging buffer (GPU→CPU readback)
    pub staging: Buffer,
    /// 최대 시퀀스 길이
    pub max_seq_len: usize,
}

impl ActivationPool {
    pub fn new(gpu: &GpuContext, max_seq_len: usize, d_model: usize, d_ff: usize,
               nheads: usize, headdim: usize, d_state: usize, chunk_size: usize,
               n_tags: usize) -> Self {
        let d_inner = d_model; // expand=1
        let ngroups = 1usize;
        let d_conv_in = d_inner + 2 * ngroups * d_state;
        let d_in_proj = d_inner + d_conv_in + nheads; // z + xBC + dt

        let nchunks = (max_seq_len + chunk_size - 1) / chunk_size;

        // 가장 큰 matmul 출력 크기 결정
        let max_wide = d_ff.max(d_in_proj).max(d_conv_in).max(2 * d_ff);
        let buf_a_size = (max_seq_len * max_wide * 4) as u64;
        let buf_b_size = (max_seq_len * d_ff.max(d_conv_in) * 4) as u64;
        let buf_cd_size = (max_seq_len * d_model * 4) as u64;

        let ssd_da_size = (nchunks * nheads * chunk_size * 4) as u64;
        let ssd_state_size = (nchunks * nheads * headdim * d_state * 4) as u64;
        let scalars_size = (max_seq_len * 4) as u64;
        let tags_size = (max_seq_len * 4) as u64;

        let mk = |label: &str, size: u64, usage: BufferUsages| -> Buffer {
            gpu.device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };

        let rw = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;

        let buf_e_size = (max_seq_len * d_ff.max(d_inner) * 4) as u64;

        let xbc_size = (max_seq_len * d_conv_in * 4) as u64;
        let ng_ds = ngroups * d_state;
        let bc_size = (max_seq_len * ng_ds * 4) as u64;
        let dt_size = (max_seq_len * nheads * 4) as u64;

        let ssd_cb_size = (nchunks * nheads * chunk_size * chunk_size * 4) as u64;

        Self {
            buf_a: mk("act_a", buf_a_size, rw),
            buf_b: mk("act_b", buf_b_size, rw),
            buf_c: mk("act_c", buf_cd_size, rw),
            buf_d: mk("act_d", buf_cd_size, rw),
            buf_e: mk("act_e", buf_e_size, rw),
            buf_f: mk("act_f", buf_cd_size, rw),
            ssd_cb: mk("ssd_cb", ssd_cb_size, rw),
            xbc: mk("xbc", xbc_size, rw),
            xbc_conv: mk("xbc_conv", xbc_size, rw),
            x_conv: mk("x_conv", buf_cd_size, rw),
            b_conv: mk("b_conv", bc_size, rw),
            c_conv: mk("c_conv", bc_size, rw),
            dt: mk("dt", dt_size, rw),
            ssd_da_cumsum: mk("ssd_da", ssd_da_size, rw),
            ssd_chunk_states: mk("ssd_cs", ssd_state_size, rw),
            ssd_prev_states: mk("ssd_ps", ssd_state_size, rw),
            scalars: mk("scalars", scalars_size, rw),
            tags: mk("tags", tags_size, rw | BufferUsages::COPY_SRC),
            staging: mk("staging", tags_size,
                        BufferUsages::MAP_READ | BufferUsages::COPY_DST),
            max_seq_len,
        }
    }
}

/// F32 데이터를 GPU storage 버퍼로 업로드
pub fn upload_f32(gpu: &GpuContext, label: &str, data: &[f32]) -> Buffer {
    let bytes: &[u8] = bytemuck::cast_slice(data);
    gpu.device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes,
        usage: BufferUsages::STORAGE,
    })
}

/// u8 데이터를 GPU storage 버퍼로 업로드 (packed2bit 등)
pub fn upload_u8(gpu: &GpuContext, label: &str, data: &[u8]) -> Buffer {
    gpu.device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some(label),
        contents: data,
        usage: BufferUsages::STORAGE,
    })
}

/// i32 데이터를 GPU storage 버퍼로 업로드
pub fn upload_i32(gpu: &GpuContext, label: &str, data: &[i32]) -> Buffer {
    let bytes: &[u8] = bytemuck::cast_slice(data);
    gpu.device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes,
        usage: BufferUsages::STORAGE,
    })
}

/// Uniform 버퍼 생성 (bytemuck Pod 타입)
pub fn upload_uniform<T: bytemuck::Pod>(gpu: &GpuContext, label: &str, data: &T) -> Buffer {
    gpu.device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(data),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    })
}
