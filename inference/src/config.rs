use serde::Deserialize;

/// BitEditor 모델 설정
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct ModelConfig {
    pub d_model: usize,
    pub n_rwkv_layers: usize,
    pub d_inner: usize,
    pub n_heads: usize,
    pub headdim: usize,
    pub d_ff: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub n_attn_heads: usize,
    pub attn_insertion_points: Vec<usize>,
    pub lora_rank: usize,
    pub vocab_size: usize,
    pub n_tags: usize,
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub pad_id: u32,
    #[serde(default = "default_bos_id")]
    pub bos_id: u32,
    #[serde(default = "default_n_iterations")]
    pub n_iterations: usize,
}

fn default_max_seq_len() -> usize { 2048 }
fn default_eps() -> f64 { 1e-6 }
fn default_bos_id() -> u32 { 1 }
fn default_n_iterations() -> usize { 3 }

impl ModelConfig {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }

    /// 임베딩 스케일
    pub fn embed_scale(&self) -> f32 {
        (self.d_model as f32).sqrt()
    }

    /// Shared attention head 차원
    pub fn attn_d_head(&self) -> usize {
        self.d_model / self.n_attn_heads
    }
}
