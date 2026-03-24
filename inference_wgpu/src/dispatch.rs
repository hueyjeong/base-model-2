//! 셰이더 dispatch 헬퍼 — 파이프라인 생성 + bind group 조립 + dispatch
//!
//! 각 셰이더별 구조체로 파이프라인/BGL을 캐시하고,
//! dispatch_xxx(encoder, buffers, params) 형태로 호출.

use wgpu::*;
use bytemuck::{Pod, Zeroable};

use crate::gpu::GpuContext;

/// 셰이더 파이프라인 + BindGroupLayout 쌍
pub struct ShaderPipeline {
    pub pipeline: ComputePipeline,
    pub bgl: BindGroupLayout,
}

/// 셰이더 모듈 → 파이프라인 생성 헬퍼
fn create_pipeline(
    gpu: &GpuContext,
    label: &str,
    wgsl: &str,
    entry: &str,
    entries: &[BindGroupLayoutEntry],
) -> ShaderPipeline {
    let shader = gpu.device.create_shader_module(ShaderModuleDescriptor {
        label: Some(label),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let bgl = gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some(&format!("{}_bgl", label)),
        entries,
    });
    let layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some(&format!("{}_layout", label)),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = gpu.device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    });
    ShaderPipeline { pipeline, bgl }
}

/// BindGroupLayoutEntry 숏컷
fn storage_ro(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_rw(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Uniform 버퍼 생성 (매 dispatch 할당)
fn make_uniform<T: Pod>(gpu: &GpuContext, data: &T) -> Buffer {
    use wgpu::util::DeviceExt;
    gpu.device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(data),
        usage: BufferUsages::UNIFORM,
    })
}

// 하위 호환용 no-op
pub fn init_uniform_pool(_gpu: &GpuContext) {}
pub fn reset_uniform_pool() {}

/// dispatch 워크그룹 수 계산 (올림 나눗셈)
fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

// ══════════════════════════════════════════════════════
// 개별 셰이더 파이프라인들
// ══════════════════════════════════════════════════════

/// 모든 파이프라인을 한 번에 생성
pub struct AllPipelines {
    pub embedding: ShaderPipeline,
    pub rms_norm: ShaderPipeline,
    pub layer_norm: ShaderPipeline,
    pub activations: ShaderPipeline,
    pub residual_add: ShaderPipeline,
    pub reverse_seq: ShaderPipeline,
    pub argmax: ShaderPipeline,
    pub matmul_f32: ShaderPipeline,
    pub matmul_ternary: ShaderPipeline,
    pub conv1d: ShaderPipeline,
    pub extract_xbc_dt: ShaderPipeline,
    pub silu_split: ShaderPipeline,
    pub swiglu: ShaderPipeline,
    pub quantize_f32: ShaderPipeline,
    pub ssd_stage1: ShaderPipeline,
    pub ssd_stage2: ShaderPipeline,
    pub ssd_stage3: ShaderPipeline,
    pub ssd_stage4a: ShaderPipeline,
    pub ssd_stage4b: ShaderPipeline,
    pub gate_norm: ShaderPipeline,
}

impl AllPipelines {
    pub fn new(gpu: &GpuContext) -> Self {
        Self {
            embedding: create_pipeline(gpu, "embedding",
                include_str!("../shaders/embedding.wgsl"), "main",
                &[storage_ro(0), storage_ro(1), storage_rw(2), uniform(3)]),

            rms_norm: create_pipeline(gpu, "rms_norm",
                include_str!("../shaders/rms_norm.wgsl"), "main",
                &[storage_ro(0), storage_ro(1), storage_rw(2), uniform(3)]),

            layer_norm: create_pipeline(gpu, "layer_norm",
                include_str!("../shaders/layer_norm.wgsl"), "main",
                &[storage_ro(0), storage_rw(1), uniform(2)]),

            activations: create_pipeline(gpu, "activations",
                include_str!("../shaders/activations.wgsl"), "main",
                &[storage_ro(0), storage_rw(1), uniform(2)]),

            residual_add: create_pipeline(gpu, "residual_add",
                include_str!("../shaders/elementwise.wgsl"), "residual_add",
                &[storage_ro(0), storage_ro(1), storage_rw(2), uniform(3)]),

            reverse_seq: create_pipeline(gpu, "reverse_seq",
                include_str!("../shaders/reverse_seq.wgsl"), "main",
                &[storage_ro(0), storage_rw(1), uniform(2)]),

            argmax: create_pipeline(gpu, "argmax",
                include_str!("../shaders/argmax.wgsl"), "main",
                &[storage_ro(0), storage_rw(1), uniform(2)]),

            matmul_f32: create_pipeline(gpu, "matmul_f32",
                include_str!("../shaders/matmul_f32.wgsl"), "main",
                &[storage_ro(0), storage_ro(1), storage_rw(2), uniform(3)]),

            matmul_ternary: create_pipeline(gpu, "matmul_ternary",
                include_str!("../shaders/matmul_ternary.wgsl"), "main",
                &[storage_ro(0), storage_ro(1), storage_rw(2), uniform(3),
                  storage_ro(4), storage_ro(5)]),

            conv1d: create_pipeline(gpu, "conv1d",
                include_str!("../shaders/conv1d.wgsl"), "main",
                &[storage_ro(0), storage_ro(1), storage_ro(2),
                  storage_rw(3), storage_rw(4), storage_rw(5), uniform(6)]),

            extract_xbc_dt: create_pipeline(gpu, "extract_xbc_dt",
                include_str!("../shaders/extract_xbc_dt.wgsl"), "main",
                &[storage_ro(0), storage_ro(1), storage_rw(2), storage_rw(3), uniform(4)]),

            silu_split: create_pipeline(gpu, "silu_split",
                include_str!("../shaders/silu_split.wgsl"), "main",
                &[storage_ro(0), storage_rw(1), storage_rw(2), storage_rw(3), uniform(4)]),

            swiglu: create_pipeline(gpu, "swiglu",
                include_str!("../shaders/swiglu.wgsl"), "main",
                &[storage_ro(0), storage_rw(1), uniform(2)]),

            quantize_f32: create_pipeline(gpu, "quantize_f32",
                include_str!("../shaders/quantize_f32.wgsl"), "main",
                &[storage_ro(0), storage_rw(1), storage_rw(2), uniform(3)]),

            ssd_stage1: create_pipeline(gpu, "ssd_stage1",
                include_str!("../shaders/ssd_stage1.wgsl"), "main",
                &[storage_ro(0), storage_ro(1), storage_rw(2), uniform(3)]),

            ssd_stage2: create_pipeline(gpu, "ssd_stage2",
                include_str!("../shaders/ssd_stage2.wgsl"), "main",
                &[storage_ro(0), storage_ro(1), storage_ro(2), storage_ro(3),
                  storage_rw(4), uniform(5)]),

            ssd_stage3: create_pipeline(gpu, "ssd_stage3",
                include_str!("../shaders/ssd_stage3.wgsl"), "main",
                &[storage_ro(0), storage_ro(1), storage_rw(2), uniform(3)]),

            ssd_stage4a: create_pipeline(gpu, "ssd_stage4a",
                include_str!("../shaders/ssd_stage4a.wgsl"), "main",
                &[storage_ro(0), storage_ro(1), storage_rw(2), uniform(3)]),

            ssd_stage4b: create_pipeline(gpu, "ssd_stage4b",
                include_str!("../shaders/ssd_stage4b.wgsl"), "main",
                &[storage_ro(0), storage_ro(1), storage_ro(2), storage_ro(3),
                  storage_ro(4), storage_ro(5), storage_ro(6), storage_rw(7), uniform(8)]),

            gate_norm: create_pipeline(gpu, "gate_norm",
                include_str!("../shaders/gate_norm.wgsl"), "main",
                &[storage_rw(0), storage_ro(1), storage_ro(2), uniform(3)]),
        }
    }
}

// ══════════════════════════════════════════════════════
// Dispatch 파라미터 structs
// ══════════════════════════════════════════════════════

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct EmbeddingParams {
    pub d_model: u32,
    pub seq_len: u32,
    pub scale: f32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct NormParams {
    pub d: u32,
    pub eps: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ActivationParams {
    pub n: u32,
    pub act_type: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ElemParams {
    pub n: u32,
    pub _pad: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ReverseParams {
    pub d_model: u32,
    pub seq_len: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ArgmaxParams {
    pub n_tags: u32,
    pub seq_len: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MatmulF32Params {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MatmulTernaryParams {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub gamma: f32,
    pub packed_stride: u32,
    pub mode: u32,
    pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Conv1dParams {
    pub seq_len: u32,
    pub d_conv_in: u32,
    pub d_conv: u32,
    pub d_inner: u32,
    pub ng_ds: u32,
    pub _pad: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ExtractXbcDtParams {
    pub seq_len: u32,
    pub d_in_proj: u32,
    pub d_inner: u32,
    pub d_conv_in: u32,
    pub nheads: u32,
    pub _pad: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SiluSplitParams {
    pub seq_len: u32,
    pub d_inner: u32,
    pub d_conv_in: u32,
    pub ng_ds: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SwigluParams {
    pub seq_len: u32,
    pub d_ff: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct QuantizeParams {
    pub d: u32,
    pub _pad: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SsdStage1Params {
    pub seq_len: u32,
    pub nheads: u32,
    pub chunk_size: u32,
    pub nchunks: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SsdStage2Params {
    pub seq_len: u32,
    pub nheads: u32,
    pub headdim: u32,
    pub d_state: u32,
    pub ngroups: u32,
    pub chunk_size: u32,
    pub nchunks: u32,
    pub d_inner: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SsdStage3Params {
    pub nheads: u32,
    pub headdim: u32,
    pub d_state: u32,
    pub chunk_size: u32,
    pub nchunks: u32,
    pub _pad: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SsdStage4aParams {
    pub seq_len: u32,
    pub nheads: u32,
    pub d_state: u32,
    pub ngroups: u32,
    pub chunk_size: u32,
    pub nchunks: u32,
    pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SsdStage4Params {
    pub seq_len: u32,
    pub nheads: u32,
    pub headdim: u32,
    pub d_state: u32,
    pub ngroups: u32,
    pub chunk_size: u32,
    pub nchunks: u32,
    pub d_inner: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GateNormParams {
    pub d_inner: u32,
    pub d_in_proj: u32,
    pub eps: f32,
    pub _pad: u32,
}

// ══════════════════════════════════════════════════════
// Dispatch 헬퍼
// ══════════════════════════════════════════════════════

/// BindGroup 생성 + compute pass에 set + dispatch
pub fn dispatch(
    gpu: &GpuContext,
    encoder: &mut CommandEncoder,
    sp: &ShaderPipeline,
    bindings: &[BindGroupEntry],
    workgroups: (u32, u32, u32),
) {
    let bg = gpu.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &sp.bgl,
        entries: bindings,
    });
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(&sp.pipeline);
    pass.set_bind_group(0, &bg, &[]);
    pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
}


/// Buffer를 BindGroupEntry로 매핑하는 헬퍼
pub fn buf_entry(binding: u32, buffer: &Buffer) -> BindGroupEntry {
    BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

// ══════════════════════════════════════════════════════
// 고수준 dispatch 함수
// ══════════════════════════════════════════════════════

pub fn dispatch_embedding(
    gpu: &GpuContext, encoder: &mut CommandEncoder, pipes: &AllPipelines,
    embedding: &Buffer, ids_buf: &Buffer, out: &Buffer,
    d_model: u32, seq_len: u32, scale: f32,
) {
    let params = make_uniform(gpu, &EmbeddingParams { d_model, seq_len, scale, _pad: 0 });
    dispatch(gpu, encoder, &pipes.embedding, &[
        buf_entry(0, embedding), buf_entry(1, ids_buf),
        buf_entry(2, out), buf_entry(3, &params),
    ], (div_ceil(seq_len * d_model, 256), 1, 1));
}

pub fn dispatch_rms_norm(
    gpu: &GpuContext, encoder: &mut CommandEncoder, pipes: &AllPipelines,
    x: &Buffer, weight: &Buffer, out: &Buffer,
    d: u32, seq_len: u32, eps: f32,
) {
    let params = make_uniform(gpu, &NormParams { d, eps });
    dispatch(gpu, encoder, &pipes.rms_norm, &[
        buf_entry(0, x), buf_entry(1, weight),
        buf_entry(2, out), buf_entry(3, &params),
    ], (seq_len, 1, 1));
}

pub fn dispatch_layer_norm(
    gpu: &GpuContext, encoder: &mut CommandEncoder, pipes: &AllPipelines,
    x: &Buffer, out: &Buffer, d: u32, seq_len: u32, eps: f32,
) {
    let params = make_uniform(gpu, &NormParams { d, eps });
    dispatch(gpu, encoder, &pipes.layer_norm, &[
        buf_entry(0, x), buf_entry(1, out), buf_entry(2, &params),
    ], (seq_len, 1, 1));
}

pub fn dispatch_residual_add(
    gpu: &GpuContext, encoder: &mut CommandEncoder, pipes: &AllPipelines,
    a: &Buffer, b: &Buffer, out: &Buffer, n: u32,
) {
    let params = make_uniform(gpu, &ElemParams { n, _pad: [0; 3] });
    dispatch(gpu, encoder, &pipes.residual_add, &[
        buf_entry(0, a), buf_entry(1, b),
        buf_entry(2, out), buf_entry(3, &params),
    ], (div_ceil(n, 256), 1, 1));
}

pub fn dispatch_reverse_seq(
    gpu: &GpuContext, encoder: &mut CommandEncoder, pipes: &AllPipelines,
    x: &Buffer, out: &Buffer, d_model: u32, seq_len: u32,
) {
    let params = make_uniform(gpu, &ReverseParams { d_model, seq_len });
    dispatch(gpu, encoder, &pipes.reverse_seq, &[
        buf_entry(0, x), buf_entry(1, out), buf_entry(2, &params),
    ], (div_ceil(seq_len * d_model, 256), 1, 1));
}

pub fn dispatch_matmul_f32(
    gpu: &GpuContext, encoder: &mut CommandEncoder, pipes: &AllPipelines,
    weight: &Buffer, x: &Buffer, out: &Buffer,
    m: u32, n: u32, k: u32,
) {
    let params = make_uniform(gpu, &MatmulF32Params { m, n, k, _pad: 0 });
    dispatch(gpu, encoder, &pipes.matmul_f32, &[
        buf_entry(0, weight), buf_entry(1, x),
        buf_entry(2, out), buf_entry(3, &params),
    ], (div_ceil(n, 32), div_ceil(m, 32), 1));
}

pub fn dispatch_matmul_ternary(
    gpu: &GpuContext, encoder: &mut CommandEncoder, pipes: &AllPipelines,
    packed_w: &Buffer, x: &Buffer, out: &Buffer,
    token_scales: &Buffer, row_sums: &Buffer,
    m: u32, n: u32, k: u32, gamma: f32, packed_stride: u32, mode: u32,
) {
    let params = make_uniform(gpu, &MatmulTernaryParams {
        m, n, k, gamma, packed_stride, mode, _pad: [0; 2],
    });
    dispatch(gpu, encoder, &pipes.matmul_ternary, &[
        buf_entry(0, packed_w), buf_entry(1, x),
        buf_entry(2, out), buf_entry(3, &params),
        buf_entry(4, token_scales), buf_entry(5, row_sums),
    ], (div_ceil(n, 32), div_ceil(m, 32), 1));
}

/// Conv1d + SiLU + x/B/C split 퓨전
pub fn dispatch_conv1d_silu_split(
    gpu: &GpuContext, encoder: &mut CommandEncoder, pipes: &AllPipelines,
    xbc: &Buffer, weight: &Buffer, bias: &Buffer,
    x_out: &Buffer, b_out: &Buffer, c_out: &Buffer,
    seq_len: u32, d_conv_in: u32, d_conv: u32, d_inner: u32, ng_ds: u32,
) {
    let params = make_uniform(gpu, &Conv1dParams {
        seq_len, d_conv_in, d_conv, d_inner, ng_ds, _pad: [0; 3],
    });
    dispatch(gpu, encoder, &pipes.conv1d, &[
        buf_entry(0, xbc), buf_entry(1, weight), buf_entry(2, bias),
        buf_entry(3, x_out), buf_entry(4, b_out), buf_entry(5, c_out),
        buf_entry(6, &params),
    ], (div_ceil(seq_len * d_conv_in, 256), 1, 1));
}

pub fn dispatch_ssd_stage1(
    gpu: &GpuContext, encoder: &mut CommandEncoder, pipes: &AllPipelines,
    dt: &Buffer, a_neg: &Buffer, da_cumsum: &Buffer,
    seq_len: u32, nheads: u32, chunk_size: u32,
) {
    let nchunks = div_ceil(seq_len, chunk_size);
    let params = make_uniform(gpu, &SsdStage1Params { seq_len, nheads, chunk_size, nchunks });
    dispatch(gpu, encoder, &pipes.ssd_stage1, &[
        buf_entry(0, dt), buf_entry(1, a_neg),
        buf_entry(2, da_cumsum), buf_entry(3, &params),
    ], (nchunks * nheads, 1, 1));
}

pub fn dispatch_ssd_stage2(
    gpu: &GpuContext, encoder: &mut CommandEncoder, pipes: &AllPipelines,
    x: &Buffer, b: &Buffer, dt: &Buffer, da_cumsum: &Buffer, chunk_states: &Buffer,
    seq_len: u32, nheads: u32, headdim: u32, d_state: u32, ngroups: u32,
    chunk_size: u32, d_inner: u32,
) {
    let nchunks = div_ceil(seq_len, chunk_size);
    let params = make_uniform(gpu, &SsdStage2Params {
        seq_len, nheads, headdim, d_state, ngroups, chunk_size, nchunks, d_inner,
    });
    dispatch(gpu, encoder, &pipes.ssd_stage2, &[
        buf_entry(0, x), buf_entry(1, b), buf_entry(2, dt),
        buf_entry(3, da_cumsum), buf_entry(4, chunk_states), buf_entry(5, &params),
    ], (nchunks * nheads * 64, 1, 1));  // 64 threads per workgroup
}

pub fn dispatch_ssd_stage3(
    gpu: &GpuContext, encoder: &mut CommandEncoder, pipes: &AllPipelines,
    chunk_states: &Buffer, da_cumsum: &Buffer, prev_states: &Buffer,
    nheads: u32, headdim: u32, d_state: u32, chunk_size: u32, seq_len: u32,
) {
    let nchunks = div_ceil(seq_len, chunk_size);
    let params = make_uniform(gpu, &SsdStage3Params {
        nheads, headdim, d_state, chunk_size, nchunks, _pad: [0; 3],
    });
    dispatch(gpu, encoder, &pipes.ssd_stage3, &[
        buf_entry(0, chunk_states), buf_entry(1, da_cumsum),
        buf_entry(2, prev_states), buf_entry(3, &params),
    ], (nheads, 1, 1));
}

/// Stage 4a: CB 행렬 사전 계산
pub fn dispatch_ssd_stage4a(
    gpu: &GpuContext, encoder: &mut CommandEncoder, pipes: &AllPipelines,
    b: &Buffer, c: &Buffer, cb: &Buffer,
    seq_len: u32, nheads: u32, d_state: u32, ngroups: u32, chunk_size: u32,
) {
    let nchunks = div_ceil(seq_len, chunk_size);
    let params = make_uniform(gpu, &SsdStage4aParams {
        seq_len, nheads, d_state, ngroups, chunk_size, nchunks, _pad: [0; 2],
    });
    let tiles_per_row = div_ceil(chunk_size, 32);
    let n_tiles = tiles_per_row * tiles_per_row;
    dispatch(gpu, encoder, &pipes.ssd_stage4a, &[
        buf_entry(0, b), buf_entry(1, c), buf_entry(2, cb), buf_entry(3, &params),
    ], (n_tiles, nchunks * nheads, 1));
}

/// Stage 4b: Score + output (CB 사전 계산 사용)
pub fn dispatch_ssd_stage4b(
    gpu: &GpuContext, encoder: &mut CommandEncoder, pipes: &AllPipelines,
    x: &Buffer, c: &Buffer, dt: &Buffer,
    da_cumsum: &Buffer, prev_states: &Buffer, d_skip: &Buffer,
    cb: &Buffer, y: &Buffer,
    seq_len: u32, nheads: u32, headdim: u32, d_state: u32, ngroups: u32,
    chunk_size: u32, d_inner: u32,
) {
    let nchunks = div_ceil(seq_len, chunk_size);
    let params = make_uniform(gpu, &SsdStage4Params {
        seq_len, nheads, headdim, d_state, ngroups, chunk_size, nchunks, d_inner,
    });
    dispatch(gpu, encoder, &pipes.ssd_stage4b, &[
        buf_entry(0, x), buf_entry(1, c), buf_entry(2, dt),
        buf_entry(3, da_cumsum), buf_entry(4, prev_states),
        buf_entry(5, d_skip), buf_entry(6, cb), buf_entry(7, y), buf_entry(8, &params),
    ], (nheads * nchunks * chunk_size, 1, 1));
}

pub fn dispatch_gate_norm(
    gpu: &GpuContext, encoder: &mut CommandEncoder, pipes: &AllPipelines,
    y_scan: &Buffer, z_proj: &Buffer, weight: &Buffer,
    d_inner: u32, d_in_proj: u32, seq_len: u32, eps: f32,
) {
    let params = make_uniform(gpu, &GateNormParams { d_inner, d_in_proj, eps, _pad: 0 });
    dispatch(gpu, encoder, &pipes.gate_norm, &[
        buf_entry(0, y_scan), buf_entry(1, z_proj),
        buf_entry(2, weight), buf_entry(3, &params),
    ], (seq_len, 1, 1));
}

pub fn dispatch_argmax(
    gpu: &GpuContext, encoder: &mut CommandEncoder, pipes: &AllPipelines,
    logits: &Buffer, tags: &Buffer, n_tags: u32, seq_len: u32,
) {
    let params = make_uniform(gpu, &ArgmaxParams { n_tags, seq_len });
    dispatch(gpu, encoder, &pipes.argmax, &[
        buf_entry(0, logits), buf_entry(1, tags), buf_entry(2, &params),
    ], (seq_len, 1, 1));
}

pub fn dispatch_quantize_f32(
    gpu: &GpuContext, encoder: &mut CommandEncoder, pipes: &AllPipelines,
    x: &Buffer, out: &Buffer, scales: &Buffer, d: u32, seq_len: u32,
) {
    let params = make_uniform(gpu, &QuantizeParams { d, _pad: [0; 3] });
    dispatch(gpu, encoder, &pipes.quantize_f32, &[
        buf_entry(0, x), buf_entry(1, out),
        buf_entry(2, scales), buf_entry(3, &params),
    ], (seq_len, 1, 1));
}

pub fn dispatch_extract_xbc_dt(
    gpu: &GpuContext, encoder: &mut CommandEncoder, pipes: &AllPipelines,
    proj: &Buffer, dt_bias: &Buffer, xbc_out: &Buffer, dt_out: &Buffer,
    seq_len: u32, d_in_proj: u32, d_inner: u32, d_conv_in: u32, nheads: u32,
) {
    let total = (seq_len * d_conv_in).max(seq_len * nheads);
    let params = make_uniform(gpu, &ExtractXbcDtParams {
        seq_len, d_in_proj, d_inner, d_conv_in, nheads, _pad: [0; 3],
    });
    dispatch(gpu, encoder, &pipes.extract_xbc_dt, &[
        buf_entry(0, proj), buf_entry(1, dt_bias),
        buf_entry(2, xbc_out), buf_entry(3, dt_out), buf_entry(4, &params),
    ], (div_ceil(total, 256), 1, 1));
}

pub fn dispatch_silu_split(
    gpu: &GpuContext, encoder: &mut CommandEncoder, pipes: &AllPipelines,
    xbc_conv: &Buffer, x_out: &Buffer, b_out: &Buffer, c_out: &Buffer,
    seq_len: u32, d_inner: u32, d_conv_in: u32, ng_ds: u32,
) {
    let params = make_uniform(gpu, &SiluSplitParams { seq_len, d_inner, d_conv_in, ng_ds });
    dispatch(gpu, encoder, &pipes.silu_split, &[
        buf_entry(0, xbc_conv), buf_entry(1, x_out),
        buf_entry(2, b_out), buf_entry(3, c_out), buf_entry(4, &params),
    ], (div_ceil(seq_len * d_conv_in, 256), 1, 1));
}

pub fn dispatch_swiglu(
    gpu: &GpuContext, encoder: &mut CommandEncoder, pipes: &AllPipelines,
    gu: &Buffer, out: &Buffer, seq_len: u32, d_ff: u32,
) {
    let params = make_uniform(gpu, &SwigluParams { seq_len, d_ff });
    dispatch(gpu, encoder, &pipes.swiglu, &[
        buf_entry(0, gu), buf_entry(1, out), buf_entry(2, &params),
    ], (div_ceil(seq_len * d_ff, 256), 1, 1));
}
