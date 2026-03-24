//! DenseEditorGpu — 전체 모델 로드 + forward 파이프라인
//!
//! BMMQ → GPU 버퍼 업로드 → compute pass 체이닝 → tags 출력

use std::collections::HashMap;
use anyhow::Result;
use wgpu::*;

use inference_common::bmmq::{self, TensorData};
use inference_common::config::DenseEditorConfig;

use crate::gpu::GpuContext;
use crate::buffers::*;
use crate::dispatch::*;

/// Packed2Bit 가중치 (GPU 저장)
pub struct TernaryWeight {
    pub packed: Buffer,       // packed u8 데이터
    pub gamma: f32,           // per-tensor scale
    pub row_sums: Buffer,     // i32[rows]
    pub rows: usize,
    pub cols: usize,
    pub packed_stride: usize,
}

/// F32 가중치 (GPU 저장)
pub struct F32Weight {
    pub data: Buffer,
    pub rows: usize,
    pub cols: usize,
}

/// BitLinear projection: no-affine LayerNorm → quantize → ternary matmul
/// (CPU 엔진과 동일: norm.weight 없음, layer_norm_no_affine 사용)
pub struct BitLinearProj {
    pub weight: TernaryWeight,
}

/// 일반 Ternary projection (LayerNorm 없음, 기존 모델용)
pub struct TernaryProj {
    pub weight: TernaryWeight,
}

/// Mamba2Block 가중치 (한 방향)
pub struct Mamba2Weights {
    pub in_proj_down: Option<F32Weight>,   // d_model → rank (F32)
    pub in_proj_up: TernaryWeight,         // rank → d_in_proj (BitLinear 또는 Ternary)
    pub in_proj_is_bitlinear: bool,        // true이면 BitLinear (no-affine LN + quant)
    pub conv1d_weight: Buffer,             // [d_conv_in, d_conv]
    pub conv1d_bias: Buffer,               // [d_conv_in]
    pub a_neg: Buffer,                     // [nheads] — 미리 -exp(A_log) 계산
    pub d_skip: Buffer,                    // [nheads]
    pub dt_bias: Buffer,                   // [nheads]
    pub norm_weight: Buffer,               // [d_inner] RMSNorm
    pub out_proj: TernaryWeight,           // d_inner → d_model
    pub out_proj_is_bitlinear: bool,
}

/// BiMamba2 가중치 (양방향)
pub struct BiMamba2Weights {
    pub fwd: Mamba2Weights,
    pub bwd: Mamba2Weights,
}

/// FusedBitNetFFN 가중치
pub struct FfnWeights {
    pub gate_up_proj: TernaryWeight,  // d_model → 2*d_ff
    pub down_proj: TernaryWeight,     // d_ff → d_model
}

/// 레이어 가중치
pub struct LayerWeights {
    pub norm1: Buffer,       // RMSNorm weight [d_model]
    pub mixing: BiMamba2Weights,
    pub norm2: Buffer,       // RMSNorm weight [d_model]
    pub ffn: FfnWeights,
}

/// GPU 모델 전체
pub struct DenseEditorGpu {
    pub config: DenseEditorConfig,
    pub embedding: Buffer,       // [vocab_size, d_model] f32
    pub embed_scale: f32,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Buffer,      // [d_model]
    pub tag_head: TernaryWeight, // d_model → n_tags (BitLinear)
    pub pool: ActivationPool,
    pub pipes: AllPipelines,
    pub is_bitlinear: bool,      // BitLinear 모드 여부
    // 더미 버퍼 (BitLinear 미사용 시 token_scales/row_sums placeholder)
    pub dummy_f32: Buffer,
}

impl DenseEditorGpu {
    /// BMMQ 파일에서 모델 로드 → GPU 버퍼 업로드
    pub fn load(gpu: &GpuContext, config: DenseEditorConfig, model_path: &str) -> Result<Self> {
        let mut tensors = bmmq::load_bmmq(model_path)?;
        let n_tensors = tensors.len();

        let d = config.d_model;
        let expand = config.mamba_expand;
        let d_inner = d * expand;
        let ds = config.mamba2_d_state;
        let hd = config.mamba2_headdim;
        let nheads = d_inner / hd;
        let chunk_size = 256;
        let in_proj_rank = if config.bitlinear_mamba { config.mamba2_in_proj_rank } else { 0 };
        let is_bitlinear = config.bitlinear_mamba;

        let embed_scale = (d as f32).sqrt();

        // Embedding
        let embedding = take_f32_buf(gpu, &mut tensors, "embedding.weight")?;

        // Layers
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let prefix = format!("layers.{}", i);
            let layer = load_layer(gpu, &mut tensors, &prefix, in_proj_rank, is_bitlinear)?;
            layers.push(layer);
        }

        // Final norm + tag head
        let final_norm = take_f32_buf(gpu, &mut tensors, "final_norm.weight")?;
        let tag_head = take_ternary(gpu, &mut tensors, "tag_head.weight")?;

        // 미사용 텐서 (norm.weight 등) 무시
        if !tensors.is_empty() {
            let remaining: Vec<&str> = tensors.keys().map(|s| s.as_str()).collect();
            eprintln!("미사용 텐서 {}개 무시: {:?}", remaining.len(),
                     &remaining[..remaining.len().min(5)]);
        }

        // 활성화 풀
        let pool = ActivationPool::new(
            gpu, config.max_seq_len, d, config.d_ff,
            nheads, hd, ds, chunk_size, config.n_tags,
        );

        let pipes = AllPipelines::new(gpu);
        init_uniform_pool(gpu);

        // 더미 버퍼 (ternary matmul에서 BitLinear 미사용 시 placeholder)
        let dummy_f32 = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("dummy"),
            size: 16,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        eprintln!("BMMQ 텐서 {}개 중 {}개 로드", n_tensors, n_tensors - tensors.len());

        Ok(Self {
            config,
            embedding,
            embed_scale,
            layers,
            final_norm,
            tag_head,
            pool,
            pipes,
            is_bitlinear,
            dummy_f32,
        })
    }

    /// Forward pass: input_ids → tags
    pub fn forward(&mut self, gpu: &GpuContext, input_ids: &[u32]) -> Result<Vec<u32>> {
        use wgpu::util::DeviceExt;
        reset_uniform_pool();

        let seq_len = input_ids.len();
        let d = self.config.d_model as u32;
        let expand = self.config.mamba_expand;
        let d_inner = (self.config.d_model * expand) as u32;
        let ds = self.config.mamba2_d_state as u32;
        let hd = self.config.mamba2_headdim as u32;
        let nh = d_inner / hd;
        let ng = self.config.mamba2_ngroups as u32;
        let d_conv = self.config.mamba_d_conv as u32;
        let d_conv_in = d_inner + 2 * ng * ds;
        let d_in_proj = d_inner + d_conv_in + nh;
        let d_ff = self.config.d_ff as u32;
        let n_tags = self.config.n_tags as u32;
        let sl = seq_len as u32;
        let chunk_size = 256u32;
        let eps = self.config.rms_norm_eps as f32;
        let in_proj_rank = if self.is_bitlinear { self.config.mamba2_in_proj_rank as u32 } else { 0 };

        // 1. input_ids → GPU 버퍼
        let ids_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("input_ids"),
            contents: bytemuck::cast_slice(input_ids),
            usage: BufferUsages::STORAGE,
        });

        let mut encoder = gpu.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("forward"),
        });

        // 2. Embedding lookup → buf_c
        dispatch_embedding(gpu, &mut encoder, &self.pipes,
            &self.embedding, &ids_buf, &self.pool.buf_c,
            d, sl, self.embed_scale);

        // 3. Layer loop
        // 버퍼 할당 규칙 (compute pass 간 동일 버퍼 read→write 전환은 OK):
        //   buf_c/d: hidden state 핑퐁 (절대 mamba2/ffn 내부에서 사용 안 함)
        //   buf_a: normed, proj, 임시
        //   buf_b: proj 임시, FFN 임시
        //   buf_e: scan 임시, reversed, FFN 임시
        //   buf_f: fwd/ffn 출력
        for (li, layer) in self.layers.iter().enumerate() {
            let (x_src, x_dst) = if li % 2 == 0 {
                (&self.pool.buf_c, &self.pool.buf_d)
            } else {
                (&self.pool.buf_d, &self.pool.buf_c)
            };

            // ── norm1(x_src) → buf_a ──
            dispatch_rms_norm(gpu, &mut encoder, &self.pipes,
                x_src, &layer.norm1, &self.pool.buf_a, d, sl, eps);

            // ── FWD direction: input=buf_a, proj=buf_b, tmp=buf_e, output=buf_f ──
            self.dispatch_mamba2_direction(gpu, &mut encoder,
                &layer.mixing.fwd,
                &self.pool.buf_a, &self.pool.buf_b, &self.pool.buf_e, &self.pool.buf_f,
                sl, d_inner, ds, hd, nh, ng, d_conv, d_conv_in, d_in_proj,
                in_proj_rank, chunk_size, eps);

            // ── reverse(buf_a) → buf_e ──
            dispatch_reverse_seq(gpu, &mut encoder, &self.pipes,
                &self.pool.buf_a, &self.pool.buf_e, d, sl);

            // ── BWD direction: input=buf_e, proj=buf_a, tmp=buf_b, output=buf_a ──
            // proj=buf_a와 output=buf_a: 서로 다른 pass에서 사용되므로 OK
            // (out_proj pass에서 buf_a는 write-only, proj는 이전 pass에서 완료)
            self.dispatch_mamba2_direction(gpu, &mut encoder,
                &layer.mixing.bwd,
                &self.pool.buf_e, &self.pool.buf_a, &self.pool.buf_b, &self.pool.buf_a,
                sl, d_inner, ds, hd, nh, ng, d_conv, d_conv_in, d_in_proj,
                in_proj_rank, chunk_size, eps);

            // ── reverse(buf_a) → buf_e ──
            dispatch_reverse_seq(gpu, &mut encoder, &self.pipes,
                &self.pool.buf_a, &self.pool.buf_e, d, sl);

            // ── x_dst = x_src + buf_f(fwd) + buf_e(reversed bwd) ──
            dispatch_residual_add(gpu, &mut encoder, &self.pipes,
                &self.pool.buf_f, &self.pool.buf_e, &self.pool.buf_a, sl * d);
            dispatch_residual_add(gpu, &mut encoder, &self.pipes,
                x_src, &self.pool.buf_a, x_dst, sl * d);

            // ── norm2(x_dst) → buf_a ──
            dispatch_rms_norm(gpu, &mut encoder, &self.pipes,
                x_dst, &layer.norm2, &self.pool.buf_a, d, sl, eps);

            // ── FFN: buf_a(input) → buf_f(output), 내부에서 buf_e, buf_b 사용 ──
            self.dispatch_ffn(gpu, &mut encoder, &layer.ffn,
                &self.pool.buf_a, &self.pool.buf_f,
                sl, d, d_ff, eps);

            // ── x_dst = x_dst + buf_f(ffn) ──
            dispatch_residual_add(gpu, &mut encoder, &self.pipes,
                x_dst, &self.pool.buf_f, &self.pool.buf_e, sl * d);
            encoder.copy_buffer_to_buffer(&self.pool.buf_e, 0, x_dst, 0, (sl * d * 4) as u64);
        }

        // 4. Final: x → norm → layernorm → tag_head → argmax
        let final_x = if self.layers.len() % 2 == 0 {
            &self.pool.buf_c
        } else {
            &self.pool.buf_d
        };

        dispatch_rms_norm(gpu, &mut encoder, &self.pipes,
            final_x, &self.final_norm, &self.pool.buf_a, d, sl, eps);

        // BitLinear tag_head: LayerNorm (no affine) → ternary matmul
        dispatch_layer_norm(gpu, &mut encoder, &self.pipes,
            &self.pool.buf_a, &self.pool.buf_b, d, sl, 1e-5);

        dispatch_matmul_ternary(gpu, &mut encoder, &self.pipes,
            &self.tag_head.packed, &self.pool.buf_b, &self.pool.buf_a,
            &self.dummy_f32, &self.tag_head.row_sums,
            self.tag_head.rows as u32, sl, self.tag_head.cols as u32,
            self.tag_head.gamma, self.tag_head.packed_stride as u32, 0);

        // argmax
        dispatch_argmax(gpu, &mut encoder, &self.pipes,
            &self.pool.buf_a, &self.pool.tags, n_tags, sl);

        // 5. Readback
        encoder.copy_buffer_to_buffer(
            &self.pool.tags, 0,
            &self.pool.staging, 0,
            (sl * 4) as u64,
        );

        gpu.queue.submit(Some(encoder.finish()));

        let slice = self.pool.staging.slice(..sl as u64 * 4);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        gpu.device.poll(Maintain::Wait);
        rx.recv()??;

        let data = slice.get_mapped_range();
        let tags: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.pool.staging.unmap();

        Ok(tags)
    }

    /// BitLinear matmul: LayerNorm → quantize → ternary matmul(mode=1, per-token scale)
    /// x_buf → [LN → quantize → scalars] → matmul → out (+ scale)
    /// norm_buf, quant_buf: 임시 [sl, in_dim], scalars_buf: [sl]
    fn dispatch_bitlinear_matmul(
        &self, gpu: &GpuContext, encoder: &mut CommandEncoder,
        w: &TernaryWeight,
        x_buf: &Buffer, norm_buf: &Buffer, quant_buf: &Buffer,
        scales_buf: &Buffer, out_buf: &Buffer,
        sl: u32, in_dim: u32,
    ) {
        // 1. LayerNorm (no affine)
        dispatch_layer_norm(gpu, encoder, &self.pipes,
            x_buf, norm_buf, in_dim, sl, 1e-5);
        // 2. Per-token quantize
        dispatch_quantize_f32(gpu, encoder, &self.pipes,
            norm_buf, quant_buf, scales_buf, in_dim, sl);
        // 3. Ternary matmul with per-token scale
        dispatch_matmul_ternary(gpu, encoder, &self.pipes,
            &w.packed, quant_buf, out_buf,
            scales_buf, &w.row_sums,
            w.rows as u32, sl, w.cols as u32,
            w.gamma, w.packed_stride as u32, 1);
    }

    /// 일반 ternary matmul (BitLinear 아님)
    fn dispatch_ternary_matmul(
        &self, gpu: &GpuContext, encoder: &mut CommandEncoder,
        w: &TernaryWeight, x_buf: &Buffer, out_buf: &Buffer, sl: u32,
    ) {
        dispatch_matmul_ternary(gpu, encoder, &self.pipes,
            &w.packed, x_buf, out_buf,
            &self.dummy_f32, &w.row_sums,
            w.rows as u32, sl, w.cols as u32,
            w.gamma, w.packed_stride as u32, 0);
    }

    /// Mamba2 단일 방향 dispatch
    /// CPU 엔진 흐름: in_proj → extract xbc/dt → conv1d → SiLU split → SSD → gate+norm → out_proj
    fn dispatch_mamba2_direction(
        &self, gpu: &GpuContext, encoder: &mut CommandEncoder,
        weights: &Mamba2Weights,
        input: &Buffer, proj_buf: &Buffer, tmp_buf: &Buffer, output: &Buffer,
        sl: u32, d_inner: u32, ds: u32, hd: u32, nh: u32, ng: u32,
        d_conv: u32, d_conv_in: u32, d_in_proj: u32, in_proj_rank: u32,
        chunk_size: u32, eps: f32,
    ) {
        let ng_ds = ng * ds;

        // 1. in_proj: input → proj_buf[sl, d_in_proj]
        if in_proj_rank > 0 {
            if let Some(ref down) = weights.in_proj_down {
                // F32 down: input → tmp_buf[sl, rank]
                dispatch_matmul_f32(gpu, encoder, &self.pipes,
                    &down.data, input, tmp_buf,
                    down.rows as u32, sl, down.cols as u32);
                // BitLinear up: tmp_buf → proj_buf (LayerNorm + quantize + matmul)
                if self.is_bitlinear {
                    self.dispatch_bitlinear_matmul(gpu, encoder,
                        &weights.in_proj_up,
                        tmp_buf, &self.pool.x_conv, &self.pool.b_conv,  // norm, quant 임시
                        &self.pool.scalars, proj_buf,
                        sl, weights.in_proj_up.cols as u32);
                } else {
                    self.dispatch_ternary_matmul(gpu, encoder,
                        &weights.in_proj_up, tmp_buf, proj_buf, sl);
                }
            }
        } else {
            if self.is_bitlinear {
                self.dispatch_bitlinear_matmul(gpu, encoder,
                    &weights.in_proj_up,
                    input, tmp_buf, &self.pool.x_conv,  // norm, quant 임시
                    &self.pool.scalars, proj_buf,
                    sl, weights.in_proj_up.cols as u32);
            } else {
                self.dispatch_ternary_matmul(gpu, encoder,
                    &weights.in_proj_up, input, proj_buf, sl);
            }
        }

        // 2. extract xBC + dt from proj
        dispatch_extract_xbc_dt(gpu, encoder, &self.pipes,
            proj_buf, &weights.dt_bias, &self.pool.xbc, &self.pool.dt,
            sl, d_in_proj, d_inner, d_conv_in, nh);

        // 3+4. conv1d + SiLU + split (퓨전)
        dispatch_conv1d_silu_split(gpu, encoder, &self.pipes,
            &self.pool.xbc, &weights.conv1d_weight, &weights.conv1d_bias,
            &self.pool.x_conv, &self.pool.b_conv, &self.pool.c_conv,
            sl, d_conv_in, d_conv, d_inner, ng_ds);

        // 5. SSD scan: 4 stages
        dispatch_ssd_stage1(gpu, encoder, &self.pipes,
            &self.pool.dt, &weights.a_neg, &self.pool.ssd_da_cumsum,
            sl, nh, chunk_size);

        dispatch_ssd_stage2(gpu, encoder, &self.pipes,
            &self.pool.x_conv, &self.pool.b_conv, &self.pool.dt,
            &self.pool.ssd_da_cumsum, &self.pool.ssd_chunk_states,
            sl, nh, hd, ds, ng, chunk_size, d_inner);

        dispatch_ssd_stage3(gpu, encoder, &self.pipes,
            &self.pool.ssd_chunk_states, &self.pool.ssd_da_cumsum,
            &self.pool.ssd_prev_states,
            nh, hd, ds, chunk_size, sl);

        dispatch_ssd_stage4(gpu, encoder, &self.pipes,
            &self.pool.x_conv, &self.pool.b_conv, &self.pool.c_conv, &self.pool.dt,
            &self.pool.ssd_da_cumsum, &self.pool.ssd_prev_states,
            &weights.d_skip, tmp_buf,  // y → tmp_buf
            sl, nh, hd, ds, ng, chunk_size, d_inner);

        // 6. gate + norm: tmp_buf = RMSNorm(tmp_buf * SiLU(z from proj_buf))
        dispatch_gate_norm(gpu, encoder, &self.pipes,
            tmp_buf, proj_buf, &weights.norm_weight,
            d_inner, d_in_proj, sl, eps);

        // 7. out_proj: tmp_buf → output
        if self.is_bitlinear {
            self.dispatch_bitlinear_matmul(gpu, encoder,
                &weights.out_proj,
                tmp_buf, &self.pool.x_conv, &self.pool.b_conv,
                &self.pool.scalars, output,
                sl, weights.out_proj.cols as u32);
        } else {
            self.dispatch_ternary_matmul(gpu, encoder,
                &weights.out_proj, tmp_buf, output, sl);
        }
    }

    /// FFN dispatch: LayerNorm → gate_up (ternary) → ReLU*gate → LayerNorm → down (ternary)
    /// input: buf_a, output: buf_f (호출자가 지정)
    /// 내부에서 buf_b, buf_e 사용
    fn dispatch_ffn(
        &self, gpu: &GpuContext, encoder: &mut CommandEncoder,
        ffn: &FfnWeights, input: &Buffer, output: &Buffer,
        sl: u32, d: u32, d_ff: u32, _eps: f32,
    ) {
        // 1. LayerNorm(input) → buf_e (BitLinear pre-norm, no affine)
        dispatch_layer_norm(gpu, encoder, &self.pipes,
            input, &self.pool.buf_e, d, sl, 1e-5);

        // 2. gate_up matmul: buf_e[sl, d] → buf_a[sl, 2*d_ff]
        dispatch_matmul_ternary(gpu, encoder, &self.pipes,
            &ffn.gate_up_proj.packed, &self.pool.buf_e, &self.pool.buf_a,
            &self.dummy_f32, &ffn.gate_up_proj.row_sums,
            ffn.gate_up_proj.rows as u32, sl, ffn.gate_up_proj.cols as u32,
            ffn.gate_up_proj.gamma, ffn.gate_up_proj.packed_stride as u32, 0);

        // 3. ReLU(gate) * up → buf_e[sl, d_ff]
        dispatch_swiglu(gpu, encoder, &self.pipes,
            &self.pool.buf_a, &self.pool.buf_e, sl, d_ff);

        // 4. LayerNorm(mid, no affine) → buf_b[sl, d_ff]
        dispatch_layer_norm(gpu, encoder, &self.pipes,
            &self.pool.buf_e, &self.pool.buf_b, d_ff, sl, 1e-5);

        // 5. down matmul: buf_b[sl, d_ff] → output[sl, d]
        dispatch_matmul_ternary(gpu, encoder, &self.pipes,
            &ffn.down_proj.packed, &self.pool.buf_b, output,
            &self.dummy_f32, &ffn.down_proj.row_sums,
            ffn.down_proj.rows as u32, sl, ffn.down_proj.cols as u32,
            ffn.down_proj.gamma, ffn.down_proj.packed_stride as u32, 0);
    }
}

// ── 헬퍼 함수 ──────────────────────────────────

/// TensorData에서 F32 추출 → GPU 버퍼
fn take_f32_buf(gpu: &GpuContext, tensors: &mut HashMap<String, TensorData>, name: &str) -> Result<Buffer> {
    let td = tensors.remove(name)
        .ok_or_else(|| anyhow::anyhow!("텐서 미발견: {}", name))?;
    match td {
        TensorData::F32 { data, .. } => Ok(upload_f32(gpu, name, &data)),
        _ => anyhow::bail!("텐서 {}가 F32 아님", name),
    }
}

/// Packed2Bit 텐서 → TernaryWeight GPU 버퍼
fn take_ternary(gpu: &GpuContext, tensors: &mut HashMap<String, TensorData>, name: &str) -> Result<TernaryWeight> {
    let td = tensors.remove(name)
        .ok_or_else(|| anyhow::anyhow!("텐서 미발견: {}", name))?;
    match td {
        TensorData::Packed2Bit { data, gamma, row_sums, rows, cols, packed_stride } => {
            Ok(TernaryWeight {
                packed: upload_u8(gpu, &format!("{}_packed", name), &data),
                gamma,
                row_sums: upload_i32(gpu, &format!("{}_rsums", name), &row_sums),
                rows,
                cols,
                packed_stride,
            })
        }
        _ => anyhow::bail!("텐서 {}가 Packed2Bit 아님", name),
    }
}

/// F32 행렬 → F32Weight GPU 버퍼
fn take_f32_weight(gpu: &GpuContext, tensors: &mut HashMap<String, TensorData>, name: &str) -> Result<F32Weight> {
    let td = tensors.remove(name)
        .ok_or_else(|| anyhow::anyhow!("텐서 미발견: {}", name))?;
    match td {
        TensorData::F32 { data, shape } => {
            let rows = shape[0];
            let cols = if shape.len() > 1 { shape[1] } else { 1 };
            Ok(F32Weight {
                data: upload_f32(gpu, name, &data),
                rows,
                cols,
            })
        }
        _ => anyhow::bail!("텐서 {}가 F32 아님", name),
    }
}

/// Mamba2Block 가중치 로드 (한 방향)
fn load_mamba2_weights(
    gpu: &GpuContext, tensors: &mut HashMap<String, TensorData>,
    prefix: &str, in_proj_rank: usize, is_bitlinear: bool,
) -> Result<Mamba2Weights> {
    let (in_proj_down, in_proj_up) = if in_proj_rank > 0 {
        let down = take_f32_weight(gpu, tensors, &format!("{}.in_proj_down.weight", prefix))?;
        let up = take_ternary(gpu, tensors, &format!("{}.in_proj_up.weight", prefix))?;
        (Some(down), up)
    } else {
        let up = take_ternary(gpu, tensors, &format!("{}.in_proj.weight", prefix))?;
        (None, up)
    };

    let a_log_data = take_f32_data(tensors, &format!("{}.A_log", prefix))?;
    let a_neg: Vec<f32> = a_log_data.iter().map(|v| -v.exp()).collect();

    let out_proj = take_ternary(gpu, tensors, &format!("{}.out_proj.weight", prefix))?;

    Ok(Mamba2Weights {
        in_proj_down,
        in_proj_up,
        in_proj_is_bitlinear: is_bitlinear,
        conv1d_weight: take_f32_buf(gpu, tensors, &format!("{}.conv1d.weight", prefix))?,
        conv1d_bias: take_f32_buf(gpu, tensors, &format!("{}.conv1d.bias", prefix))?,
        a_neg: upload_f32(gpu, &format!("{}.a_neg", prefix), &a_neg),
        d_skip: take_f32_buf(gpu, tensors, &format!("{}.D", prefix))?,
        dt_bias: take_f32_buf(gpu, tensors, &format!("{}.dt_bias", prefix))?,
        norm_weight: take_f32_buf(gpu, tensors, &format!("{}.norm.weight", prefix))?,
        out_proj,
        out_proj_is_bitlinear: is_bitlinear,
    })
}

/// 레이어 전체 로드
fn load_layer(
    gpu: &GpuContext, tensors: &mut HashMap<String, TensorData>,
    prefix: &str, in_proj_rank: usize, is_bitlinear: bool,
) -> Result<LayerWeights> {
    let norm1 = take_f32_buf(gpu, tensors, &format!("{}.norm1.weight", prefix))?;
    let fwd = load_mamba2_weights(gpu, tensors, &format!("{}.mixing.fwd", prefix), in_proj_rank, is_bitlinear)?;
    let bwd = load_mamba2_weights(gpu, tensors, &format!("{}.mixing.bwd", prefix), in_proj_rank, is_bitlinear)?;
    let norm2 = take_f32_buf(gpu, tensors, &format!("{}.norm2.weight", prefix))?;

    let gate_up = take_ternary(gpu, tensors, &format!("{}.ffn.gate_up_proj.weight", prefix))?;
    let down = take_ternary(gpu, tensors, &format!("{}.ffn.down_proj.weight", prefix))?;

    Ok(LayerWeights {
        norm1,
        mixing: BiMamba2Weights { fwd, bwd },
        norm2,
        ffn: FfnWeights { gate_up_proj: gate_up, down_proj: down },
    })
}

/// F32 텐서 데이터를 Vec<f32>로 추출 (GPU 미업로드, CPU에서 전처리용)
fn take_f32_data(tensors: &mut HashMap<String, TensorData>, name: &str) -> Result<Vec<f32>> {
    let td = tensors.remove(name)
        .ok_or_else(|| anyhow::anyhow!("텐서 미발견: {}", name))?;
    match td {
        TensorData::F32 { data, .. } => Ok(data),
        _ => anyhow::bail!("텐서 {}가 F32 아님", name),
    }
}
