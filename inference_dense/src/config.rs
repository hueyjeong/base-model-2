//! DenseEditor 설정 — JSON 파싱

use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
pub struct DenseEditorConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub vocab_size: usize,
    pub n_tags: usize,
    pub max_seq_len: usize,
    pub dropout: f64,
    pub rms_norm_eps: f64,
    pub pad_id: usize,
    pub bos_id: usize,
    pub mixing_type: String,

    // Mixing 공통
    pub n_heads: usize,
    pub headdim: usize,

    // Mamba
    #[serde(default = "default_16")]
    pub mamba_d_state: usize,
    #[serde(default = "default_4")]
    pub mamba_d_conv: usize,
    #[serde(default = "default_2")]
    pub mamba_expand: usize,

    // TCN
    #[serde(default = "default_7")]
    pub tcn_kernel_size: usize,
    #[serde(default = "default_6")]
    pub tcn_n_dilations: usize,

    // RetNet
    #[serde(default = "default_gamma_min")]
    pub retnet_gamma_min: f64,
    #[serde(default = "default_gamma_max")]
    pub retnet_gamma_max: f64,
}

fn default_16() -> usize { 16 }
fn default_4() -> usize { 4 }
fn default_2() -> usize { 2 }
fn default_7() -> usize { 7 }
fn default_6() -> usize { 6 }
fn default_gamma_min() -> f64 { 0.8 }
fn default_gamma_max() -> f64 { 0.999 }

impl DenseEditorConfig {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }

    pub fn d_inner(&self) -> usize {
        self.d_model * self.mamba_expand
    }

    pub fn dt_rank(&self) -> usize {
        (self.d_model / 16).max(1)
    }
}
