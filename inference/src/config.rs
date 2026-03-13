use serde::Deserialize;

/// BitMamba Seq2Seq 모델 설정
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub d_model: usize,
    pub d_inner: usize,
    pub d_ff: usize,
    pub n_encoder_layers: usize,
    pub n_decoder_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub dt_rank: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub vocab_size: usize,
    pub bos_id: u32,
    pub use_copy_gate: bool,
    pub mamba_version: usize,
    pub headdim: usize,
    pub ngroups: usize,
    pub chunk_size: usize,
    #[serde(default = "default_true")]
    pub tie_embeddings: bool,
    #[serde(default = "default_true")]
    pub tie_lm_head: bool,
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,
    #[serde(default = "default_dropout")]
    pub dropout: f64,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub pad_id: u32,
}

fn default_true() -> bool { true }
fn default_max_seq_len() -> usize { 512 }
fn default_dropout() -> f64 { 0.1 }
fn default_eps() -> f64 { 1e-6 }

impl ModelConfig {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }

    /// Mamba-2 head 수 (d_inner / headdim)
    pub fn nheads(&self) -> usize {
        self.d_inner / self.headdim
    }

    /// Cross-attention head 당 차원
    pub fn d_head(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// 임베딩 스케일
    pub fn embed_scale(&self) -> f32 {
        (self.d_model as f32).sqrt()
    }
}
