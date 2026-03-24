//! 셰이더 dispatch 헬퍼 — push constants + single compute pass
//!
//! uniform 버퍼 할당 완전 제거: params를 push constants로 커맨드 버퍼에 직접 기록.
//! forward 1회당 GPU 메모리 할당 0회.

use wgpu::*;
use bytemuck::{Pod, Zeroable};

use crate::gpu::GpuContext;

/// 셰이더 파이프라인 + BindGroupLayout 쌍
pub struct ShaderPipeline {
    pub pipeline: ComputePipeline,
    pub bgl: BindGroupLayout,
}

/// 셰이더 모듈 → 파이프라인 생성 (push constants 지원)
fn create_pipeline(
    gpu: &GpuContext,
    label: &str,
    wgsl: &str,
    entry: &str,
    entries: &[BindGroupLayoutEntry],
    push_constant_size: u32,
) -> ShaderPipeline {
    let shader = gpu.device.create_shader_module(ShaderModuleDescriptor {
        label: Some(label),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let bgl = gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some(&format!("{}_bgl", label)),
        entries,
    });

    let pc_ranges = if push_constant_size > 0 {
        vec![PushConstantRange {
            stages: ShaderStages::COMPUTE,
            range: 0..push_constant_size,
        }]
    } else {
        vec![]
    };

    let layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some(&format!("{}_layout", label)),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &pc_ranges,
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

fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

// ══════════════════════════════════════════════════════
// 파이프라인
// ══════════════════════════════════════════════════════

pub struct AllPipelines {
    pub embedding: ShaderPipeline,
    pub rms_norm: ShaderPipeline,
    pub layer_norm: ShaderPipeline,
    pub residual_add: ShaderPipeline,
    pub add_inplace: ShaderPipeline,
    pub reverse_seq: ShaderPipeline,
    pub argmax: ShaderPipeline,
    pub matmul_f32: ShaderPipeline,
    pub unpack_ternary: ShaderPipeline,
    pub matmul_f16w: ShaderPipeline,
    pub matmul_ternary: ShaderPipeline,
    pub conv1d: ShaderPipeline,
    pub extract_xbc_dt: ShaderPipeline,
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
        let elem_src = include_str!("../shaders/elementwise.wgsl");
        // elementwise: a(ro:0) + b(ro:1) + out(rw:2), push_constant로 params
        let elem_bgl = &[storage_ro(0), storage_ro(1), storage_rw(2)];
        let elem_pc = std::mem::size_of::<ElemParams>() as u32;

        Self {
            embedding: create_pipeline(gpu, "embedding",
                include_str!("../shaders/embedding.wgsl"), "main",
                &[storage_ro(0), storage_ro(1), storage_rw(2)],
                std::mem::size_of::<EmbeddingParams>() as u32),

            rms_norm: create_pipeline(gpu, "rms_norm",
                include_str!("../shaders/rms_norm.wgsl"), "main",
                &[storage_ro(0), storage_ro(1), storage_rw(2)],
                std::mem::size_of::<NormParams>() as u32),

            layer_norm: create_pipeline(gpu, "layer_norm",
                include_str!("../shaders/layer_norm.wgsl"), "main",
                &[storage_ro(0), storage_rw(1)],
                std::mem::size_of::<NormParams>() as u32),

            residual_add: create_pipeline(gpu, "residual_add",
                elem_src, "residual_add", elem_bgl, elem_pc),

            add_inplace: create_pipeline(gpu, "add_inplace",
                elem_src, "add_inplace", elem_bgl, elem_pc),

            reverse_seq: create_pipeline(gpu, "reverse_seq",
                include_str!("../shaders/reverse_seq.wgsl"), "main",
                &[storage_ro(0), storage_rw(1)],
                std::mem::size_of::<ReverseParams>() as u32),

            argmax: create_pipeline(gpu, "argmax",
                include_str!("../shaders/argmax.wgsl"), "main",
                &[storage_ro(0), storage_rw(1)],
                std::mem::size_of::<ArgmaxParams>() as u32),

            matmul_f32: create_pipeline(gpu, "matmul_f32",
                include_str!("../shaders/matmul_f32.wgsl"), "main",
                &[storage_ro(0), storage_ro(1), storage_rw(2), storage_ro(3)],
                std::mem::size_of::<MatmulF32Params>() as u32),

            unpack_ternary: create_pipeline(gpu, "unpack_ternary",
                include_str!("../shaders/unpack_ternary.wgsl"), "main",
                &[storage_ro(0), storage_rw(1)],
                std::mem::size_of::<UnpackTernaryParams>() as u32),

            // f16 weight matmul: weight(f16), x(f32), out(f32), token_scales(f32)
            matmul_f16w: create_pipeline(gpu, "matmul_f16w",
                include_str!("../shaders/matmul_f16w.wgsl"), "main",
                &[storage_ro(0), storage_ro(1), storage_rw(2), storage_ro(3)],
                std::mem::size_of::<MatmulF32Params>() as u32),  // same params as f32

            // matmul_ternary: packed2bit (폴백)
            matmul_ternary: create_pipeline(gpu, "matmul_ternary",
                include_str!("../shaders/matmul_ternary.wgsl"), "main",
                &[storage_ro(0), storage_ro(1), storage_rw(2),
                  storage_ro(3), storage_ro(4)],
                std::mem::size_of::<MatmulTernaryParams>() as u32),

            conv1d: create_pipeline(gpu, "conv1d",
                include_str!("../shaders/conv1d.wgsl"), "main",
                &[storage_ro(0), storage_ro(1), storage_ro(2),
                  storage_rw(3), storage_rw(4), storage_rw(5)],
                std::mem::size_of::<Conv1dParams>() as u32),

            extract_xbc_dt: create_pipeline(gpu, "extract_xbc_dt",
                include_str!("../shaders/extract_xbc_dt.wgsl"), "main",
                &[storage_ro(0), storage_ro(1), storage_rw(2), storage_rw(3)],
                std::mem::size_of::<ExtractXbcDtParams>() as u32),

            swiglu: create_pipeline(gpu, "swiglu",
                include_str!("../shaders/swiglu.wgsl"), "main",
                &[storage_ro(0), storage_rw(1)],
                std::mem::size_of::<SwigluParams>() as u32),

            quantize_f32: create_pipeline(gpu, "quantize_f32",
                include_str!("../shaders/quantize_f32.wgsl"), "main",
                &[storage_ro(0), storage_rw(1), storage_rw(2)],
                std::mem::size_of::<QuantizeParams>() as u32),

            ssd_stage1: create_pipeline(gpu, "ssd_stage1",
                include_str!("../shaders/ssd_stage1.wgsl"), "main",
                &[storage_ro(0), storage_ro(1), storage_rw(2)],
                std::mem::size_of::<SsdStage1Params>() as u32),

            ssd_stage2: create_pipeline(gpu, "ssd_stage2",
                include_str!("../shaders/ssd_stage2.wgsl"), "main",
                &[storage_ro(0), storage_ro(1), storage_ro(2), storage_ro(3),
                  storage_rw(4)],
                std::mem::size_of::<SsdStage2Params>() as u32),

            ssd_stage3: create_pipeline(gpu, "ssd_stage3",
                include_str!("../shaders/ssd_stage3.wgsl"), "main",
                &[storage_ro(0), storage_ro(1), storage_rw(2)],
                std::mem::size_of::<SsdStage3Params>() as u32),

            ssd_stage4a: create_pipeline(gpu, "ssd_stage4a",
                include_str!("../shaders/ssd_stage4a.wgsl"), "main",
                &[storage_ro(0), storage_ro(1), storage_rw(2)],
                std::mem::size_of::<SsdStage4aParams>() as u32),

            ssd_stage4b: create_pipeline(gpu, "ssd_stage4b",
                include_str!("../shaders/ssd_stage4b.wgsl"), "main",
                &[storage_ro(0), storage_ro(1), storage_ro(2), storage_ro(3),
                  storage_ro(4), storage_ro(5), storage_ro(6), storage_rw(7)],
                std::mem::size_of::<SsdStage4Params>() as u32),

            gate_norm: create_pipeline(gpu, "gate_norm",
                include_str!("../shaders/gate_norm.wgsl"), "main",
                &[storage_rw(0), storage_ro(1), storage_ro(2)],
                std::mem::size_of::<GateNormParams>() as u32),
        }
    }
}

// ══════════════════════════════════════════════════════
// Dispatch 파라미터 structs
// ══════════════════════════════════════════════════════

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
pub struct EmbeddingParams { pub d_model: u32, pub seq_len: u32, pub scale: f32, pub _pad: u32 }

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
pub struct NormParams { pub d: u32, pub eps: f32 }

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
pub struct ElemParams { pub n: u32, pub _pad: [u32; 3] }

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
pub struct ReverseParams { pub d_model: u32, pub seq_len: u32 }

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
pub struct ArgmaxParams { pub n_tags: u32, pub seq_len: u32 }

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
pub struct MatmulF32Params { pub m: u32, pub n: u32, pub k: u32, pub mode: u32 }

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
pub struct UnpackTernaryParams { pub rows: u32, pub cols: u32, pub packed_stride: u32, pub gamma: f32 }

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
pub struct MatmulTernaryParams {
    pub m: u32, pub n: u32, pub k: u32, pub gamma: f32,
    pub packed_stride: u32, pub mode: u32, pub _pad: [u32; 2],
}

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
pub struct Conv1dParams {
    pub seq_len: u32, pub d_conv_in: u32, pub d_conv: u32, pub d_inner: u32,
    pub ng_ds: u32, pub _pad: [u32; 3],
}

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
pub struct ExtractXbcDtParams {
    pub seq_len: u32, pub d_in_proj: u32, pub d_inner: u32, pub d_conv_in: u32,
    pub nheads: u32, pub _pad: [u32; 3],
}

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
pub struct SwigluParams { pub seq_len: u32, pub d_ff: u32 }

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
pub struct QuantizeParams { pub d: u32, pub _pad: [u32; 3] }

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
pub struct SsdStage1Params { pub seq_len: u32, pub nheads: u32, pub chunk_size: u32, pub nchunks: u32 }

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
pub struct SsdStage2Params {
    pub seq_len: u32, pub nheads: u32, pub headdim: u32, pub d_state: u32,
    pub ngroups: u32, pub chunk_size: u32, pub nchunks: u32, pub d_inner: u32,
}

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
pub struct SsdStage3Params {
    pub nheads: u32, pub headdim: u32, pub d_state: u32, pub chunk_size: u32,
    pub nchunks: u32, pub _pad: [u32; 3],
}

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
pub struct SsdStage4aParams {
    pub seq_len: u32, pub nheads: u32, pub d_state: u32, pub ngroups: u32,
    pub chunk_size: u32, pub nchunks: u32, pub _pad: [u32; 2],
}

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
pub struct SsdStage4Params {
    pub seq_len: u32, pub nheads: u32, pub headdim: u32, pub d_state: u32,
    pub ngroups: u32, pub chunk_size: u32, pub nchunks: u32, pub d_inner: u32,
}

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
pub struct GateNormParams { pub d_inner: u32, pub d_in_proj: u32, pub eps: f32, pub _pad: u32 }

// ══════════════════════════════════════════════════════
// Dispatch 헬퍼
// ══════════════════════════════════════════════════════

/// dispatch 횟수 카운터 (디버그용)
static DISPATCH_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub fn reset_dispatch_count() { DISPATCH_COUNT.store(0, std::sync::atomic::Ordering::Relaxed); }
pub fn get_dispatch_count() -> u32 { DISPATCH_COUNT.load(std::sync::atomic::Ordering::Relaxed) }

/// Push constants 기반 dispatch (GPU 메모리 할당 0회)
fn dispatch_pc<'a, T: Pod>(
    pass: &mut ComputePass<'a>,
    gpu: &GpuContext,
    sp: &ShaderPipeline,
    bindings: &[BindGroupEntry<'_>],
    params: &T,
    workgroups: (u32, u32, u32),
) {
    DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let bg = gpu.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &sp.bgl,
        entries: bindings,
    });
    pass.set_pipeline(&sp.pipeline);
    pass.set_bind_group(0, &bg, &[]);
    pass.set_push_constants(0, bytemuck::bytes_of(params));
    pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
}

pub fn buf_entry(binding: u32, buffer: &Buffer) -> BindGroupEntry<'_> {
    BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

// ══════════════════════════════════════════════════════
// 고수준 dispatch 함수
// ══════════════════════════════════════════════════════

pub fn dispatch_embedding<'a>(
    pass: &mut ComputePass<'a>, gpu: &GpuContext, pipes: &AllPipelines,
    embedding: &Buffer, ids_buf: &Buffer, out: &Buffer,
    d_model: u32, seq_len: u32, scale: f32,
) {
    dispatch_pc(pass, gpu, &pipes.embedding, &[
        buf_entry(0, embedding), buf_entry(1, ids_buf), buf_entry(2, out),
    ], &EmbeddingParams { d_model, seq_len, scale, _pad: 0 },
    (div_ceil(seq_len * d_model, 256), 1, 1));
}

pub fn dispatch_rms_norm<'a>(
    pass: &mut ComputePass<'a>, gpu: &GpuContext, pipes: &AllPipelines,
    x: &Buffer, weight: &Buffer, out: &Buffer,
    d: u32, seq_len: u32, eps: f32,
) {
    dispatch_pc(pass, gpu, &pipes.rms_norm, &[
        buf_entry(0, x), buf_entry(1, weight), buf_entry(2, out),
    ], &NormParams { d, eps }, (seq_len, 1, 1));
}

pub fn dispatch_layer_norm<'a>(
    pass: &mut ComputePass<'a>, gpu: &GpuContext, pipes: &AllPipelines,
    x: &Buffer, out: &Buffer, d: u32, seq_len: u32, eps: f32,
) {
    dispatch_pc(pass, gpu, &pipes.layer_norm, &[
        buf_entry(0, x), buf_entry(1, out),
    ], &NormParams { d, eps }, (seq_len, 1, 1));
}

pub fn dispatch_residual_add<'a>(
    pass: &mut ComputePass<'a>, gpu: &GpuContext, pipes: &AllPipelines,
    a: &Buffer, b: &Buffer, out: &Buffer, n: u32,
) {
    dispatch_pc(pass, gpu, &pipes.residual_add, &[
        buf_entry(0, a), buf_entry(1, b), buf_entry(2, out),
    ], &ElemParams { n, _pad: [0; 3] }, (div_ceil(n, 256), 1, 1));
}

pub fn dispatch_add_inplace<'a>(
    pass: &mut ComputePass<'a>, gpu: &GpuContext, pipes: &AllPipelines,
    a: &Buffer, out: &Buffer, dummy: &Buffer, n: u32,
) {
    dispatch_pc(pass, gpu, &pipes.add_inplace, &[
        buf_entry(0, a), buf_entry(1, dummy), buf_entry(2, out),
    ], &ElemParams { n, _pad: [0; 3] }, (div_ceil(n, 256), 1, 1));
}

pub fn dispatch_reverse_seq<'a>(
    pass: &mut ComputePass<'a>, gpu: &GpuContext, pipes: &AllPipelines,
    x: &Buffer, out: &Buffer, d_model: u32, seq_len: u32,
) {
    dispatch_pc(pass, gpu, &pipes.reverse_seq, &[
        buf_entry(0, x), buf_entry(1, out),
    ], &ReverseParams { d_model, seq_len },
    (div_ceil(seq_len * d_model, 256), 1, 1));
}

pub fn dispatch_matmul_f32<'a>(
    pass: &mut ComputePass<'a>, gpu: &GpuContext, pipes: &AllPipelines,
    weight: &Buffer, x: &Buffer, out: &Buffer,
    m: u32, n: u32, k: u32,
) {
    dispatch_pc(pass, gpu, &pipes.matmul_f32, &[
        buf_entry(0, weight), buf_entry(1, x), buf_entry(2, out),
        buf_entry(3, &gpu.device.create_buffer(&BufferDescriptor {
            label: None, size: 16, usage: BufferUsages::STORAGE, mapped_at_creation: false,
        })),
    ], &MatmulF32Params { m, n, k, mode: 0 },
    (div_ceil(n, 64), div_ceil(m, 64), 1));
}

pub fn dispatch_matmul_ternary<'a>(
    pass: &mut ComputePass<'a>, gpu: &GpuContext, pipes: &AllPipelines,
    packed_w: &Buffer, x: &Buffer, out: &Buffer,
    token_scales: &Buffer, row_sums: &Buffer,
    m: u32, n: u32, k: u32, gamma: f32, packed_stride: u32, mode: u32,
) {
    // binding 번호: packed_w=0, x=1, out=2, token_scales=3, row_sums=4
    dispatch_pc(pass, gpu, &pipes.matmul_ternary, &[
        buf_entry(0, packed_w), buf_entry(1, x), buf_entry(2, out),
        buf_entry(3, token_scales), buf_entry(4, row_sums),
    ], &MatmulTernaryParams { m, n, k, gamma, packed_stride, mode, _pad: [0; 2] },
    (div_ceil(n, 32), div_ceil(m, 32), 1));
}

/// GPU에서 packed2bit → f16 언팩 (모델 로드 시 1회)
pub fn dispatch_unpack_ternary(
    gpu: &GpuContext, pipes: &AllPipelines,
    packed: &Buffer, out: &Buffer,
    rows: u32, cols: u32, packed_stride: u32, gamma: f32,
) {
    let mut encoder = gpu.device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("unpack"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("unpack"), timestamp_writes: None,
        });
        let cols_pairs = div_ceil(cols, 2);
        dispatch_pc(&mut pass, gpu, &pipes.unpack_ternary, &[
            buf_entry(0, packed), buf_entry(1, out),
        ], &UnpackTernaryParams { rows, cols, packed_stride, gamma },
        (div_ceil(rows * cols_pairs, 256), 1, 1));
    }
    gpu.queue.submit(Some(encoder.finish()));
    gpu.device.poll(Maintain::Wait);
}

/// f16 weight matmul: weight_f16[M,K] × x_f32[N,K] → out_f32[N,M]
pub fn dispatch_matmul_f16w<'a>(
    pass: &mut ComputePass<'a>, gpu: &GpuContext, pipes: &AllPipelines,
    weight_f16: &Buffer, x: &Buffer, out: &Buffer, token_scales: &Buffer,
    m: u32, n: u32, k: u32, mode: u32,
) {
    dispatch_pc(pass, gpu, &pipes.matmul_f16w, &[
        buf_entry(0, weight_f16), buf_entry(1, x), buf_entry(2, out),
        buf_entry(3, token_scales),
    ], &MatmulF32Params { m, n, k, mode },
    (div_ceil(n, 64), div_ceil(m, 64), 1));
}

pub fn dispatch_conv1d_silu_split<'a>(
    pass: &mut ComputePass<'a>, gpu: &GpuContext, pipes: &AllPipelines,
    xbc: &Buffer, weight: &Buffer, bias: &Buffer,
    x_out: &Buffer, b_out: &Buffer, c_out: &Buffer,
    seq_len: u32, d_conv_in: u32, d_conv: u32, d_inner: u32, ng_ds: u32,
) {
    dispatch_pc(pass, gpu, &pipes.conv1d, &[
        buf_entry(0, xbc), buf_entry(1, weight), buf_entry(2, bias),
        buf_entry(3, x_out), buf_entry(4, b_out), buf_entry(5, c_out),
    ], &Conv1dParams { seq_len, d_conv_in, d_conv, d_inner, ng_ds, _pad: [0; 3] },
    (div_ceil(seq_len * d_conv_in, 256), 1, 1));
}

pub fn dispatch_ssd_stage1<'a>(
    pass: &mut ComputePass<'a>, gpu: &GpuContext, pipes: &AllPipelines,
    dt: &Buffer, a_neg: &Buffer, da_cumsum: &Buffer,
    seq_len: u32, nheads: u32, chunk_size: u32,
) {
    let nchunks = div_ceil(seq_len, chunk_size);
    dispatch_pc(pass, gpu, &pipes.ssd_stage1, &[
        buf_entry(0, dt), buf_entry(1, a_neg), buf_entry(2, da_cumsum),
    ], &SsdStage1Params { seq_len, nheads, chunk_size, nchunks },
    (nchunks * nheads, 1, 1));
}

pub fn dispatch_ssd_stage2<'a>(
    pass: &mut ComputePass<'a>, gpu: &GpuContext, pipes: &AllPipelines,
    x: &Buffer, b: &Buffer, dt: &Buffer, da_cumsum: &Buffer, chunk_states: &Buffer,
    seq_len: u32, nheads: u32, headdim: u32, d_state: u32, ngroups: u32,
    chunk_size: u32, d_inner: u32,
) {
    let nchunks = div_ceil(seq_len, chunk_size);
    dispatch_pc(pass, gpu, &pipes.ssd_stage2, &[
        buf_entry(0, x), buf_entry(1, b), buf_entry(2, dt),
        buf_entry(3, da_cumsum), buf_entry(4, chunk_states),
    ], &SsdStage2Params { seq_len, nheads, headdim, d_state, ngroups, chunk_size, nchunks, d_inner },
    (nchunks * nheads * 64, 1, 1));
}

pub fn dispatch_ssd_stage3<'a>(
    pass: &mut ComputePass<'a>, gpu: &GpuContext, pipes: &AllPipelines,
    chunk_states: &Buffer, da_cumsum: &Buffer, prev_states: &Buffer,
    nheads: u32, headdim: u32, d_state: u32, chunk_size: u32, seq_len: u32,
) {
    let nchunks = div_ceil(seq_len, chunk_size);
    dispatch_pc(pass, gpu, &pipes.ssd_stage3, &[
        buf_entry(0, chunk_states), buf_entry(1, da_cumsum), buf_entry(2, prev_states),
    ], &SsdStage3Params { nheads, headdim, d_state, chunk_size, nchunks, _pad: [0; 3] },
    (nheads, 1, 1));
}

pub fn dispatch_ssd_stage4a<'a>(
    pass: &mut ComputePass<'a>, gpu: &GpuContext, pipes: &AllPipelines,
    b: &Buffer, c: &Buffer, cb: &Buffer,
    seq_len: u32, nheads: u32, d_state: u32, ngroups: u32, chunk_size: u32,
) {
    let nchunks = div_ceil(seq_len, chunk_size);
    let tiles_per_row = div_ceil(chunk_size, 32);
    let n_tiles = tiles_per_row * tiles_per_row;
    dispatch_pc(pass, gpu, &pipes.ssd_stage4a, &[
        buf_entry(0, b), buf_entry(1, c), buf_entry(2, cb),
    ], &SsdStage4aParams { seq_len, nheads, d_state, ngroups, chunk_size, nchunks, _pad: [0; 2] },
    (n_tiles, nchunks * nheads, 1));
}

pub fn dispatch_ssd_stage4b<'a>(
    pass: &mut ComputePass<'a>, gpu: &GpuContext, pipes: &AllPipelines,
    x: &Buffer, c: &Buffer, dt: &Buffer,
    da_cumsum: &Buffer, prev_states: &Buffer, d_skip: &Buffer,
    cb: &Buffer, y: &Buffer,
    seq_len: u32, nheads: u32, headdim: u32, d_state: u32, ngroups: u32,
    chunk_size: u32, d_inner: u32,
) {
    let nchunks = div_ceil(seq_len, chunk_size);
    dispatch_pc(pass, gpu, &pipes.ssd_stage4b, &[
        buf_entry(0, x), buf_entry(1, c), buf_entry(2, dt),
        buf_entry(3, da_cumsum), buf_entry(4, prev_states),
        buf_entry(5, d_skip), buf_entry(6, cb), buf_entry(7, y),
    ], &SsdStage4Params { seq_len, nheads, headdim, d_state, ngroups, chunk_size, nchunks, d_inner },
    (nheads * nchunks * chunk_size, 1, 1));
}

pub fn dispatch_gate_norm<'a>(
    pass: &mut ComputePass<'a>, gpu: &GpuContext, pipes: &AllPipelines,
    y_scan: &Buffer, z_proj: &Buffer, weight: &Buffer,
    d_inner: u32, d_in_proj: u32, seq_len: u32, eps: f32,
) {
    dispatch_pc(pass, gpu, &pipes.gate_norm, &[
        buf_entry(0, y_scan), buf_entry(1, z_proj), buf_entry(2, weight),
    ], &GateNormParams { d_inner, d_in_proj, eps, _pad: 0 },
    (seq_len, 1, 1));
}

pub fn dispatch_argmax<'a>(
    pass: &mut ComputePass<'a>, gpu: &GpuContext, pipes: &AllPipelines,
    logits: &Buffer, tags: &Buffer, n_tags: u32, seq_len: u32,
) {
    dispatch_pc(pass, gpu, &pipes.argmax, &[
        buf_entry(0, logits), buf_entry(1, tags),
    ], &ArgmaxParams { n_tags, seq_len }, (seq_len, 1, 1));
}

pub fn dispatch_quantize_f32<'a>(
    pass: &mut ComputePass<'a>, gpu: &GpuContext, pipes: &AllPipelines,
    x: &Buffer, out: &Buffer, scales: &Buffer, d: u32, seq_len: u32,
) {
    dispatch_pc(pass, gpu, &pipes.quantize_f32, &[
        buf_entry(0, x), buf_entry(1, out), buf_entry(2, scales),
    ], &QuantizeParams { d, _pad: [0; 3] }, (seq_len, 1, 1));
}

pub fn dispatch_extract_xbc_dt<'a>(
    pass: &mut ComputePass<'a>, gpu: &GpuContext, pipes: &AllPipelines,
    proj: &Buffer, dt_bias: &Buffer, xbc_out: &Buffer, dt_out: &Buffer,
    seq_len: u32, d_in_proj: u32, d_inner: u32, d_conv_in: u32, nheads: u32,
) {
    let total = (seq_len * d_conv_in).max(seq_len * nheads);
    dispatch_pc(pass, gpu, &pipes.extract_xbc_dt, &[
        buf_entry(0, proj), buf_entry(1, dt_bias), buf_entry(2, xbc_out), buf_entry(3, dt_out),
    ], &ExtractXbcDtParams { seq_len, d_in_proj, d_inner, d_conv_in, nheads, _pad: [0; 3] },
    (div_ceil(total, 256), 1, 1));
}

pub fn dispatch_swiglu<'a>(
    pass: &mut ComputePass<'a>, gpu: &GpuContext, pipes: &AllPipelines,
    gu: &Buffer, out: &Buffer, seq_len: u32, d_ff: u32,
) {
    dispatch_pc(pass, gpu, &pipes.swiglu, &[
        buf_entry(0, gu), buf_entry(1, out),
    ], &SwigluParams { seq_len, d_ff },
    (div_ceil(seq_len * d_ff, 256), 1, 1));
}
