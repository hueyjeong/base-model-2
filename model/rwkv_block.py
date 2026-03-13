"""RWKV-6 TimeMix 블록

RWKV-6 (Eagle) 아키텍처의 핵심 모듈:
- 행렬 값 상태(matrix-valued state)를 가진 선형 RNN
- 상태 업데이트: state = diag(exp(w)) @ state + k ⊗ v
- 출력: output = state @ r
- 데이터 의존 감쇄(w): LoRA로 입력에서 감쇄율 생성

GPU 학습:
- flash-linear-attention (fla) 라이브러리 감지 → chunk_rwkv6 사용
- 폴백: Python 순차 루프
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.bitlinear import BitLinear

# fla 라이브러리 감지 (GPU fused recurrent 가속)
_FLA_AVAILABLE = False
_fused_recurrent_rwkv6 = None

try:
    from fla.ops.rwkv6 import fused_recurrent_rwkv6 as _fused_recurrent_rwkv6_fn
    _FLA_AVAILABLE = True
    _fused_recurrent_rwkv6 = _fused_recurrent_rwkv6_fn
except ImportError:
    pass


def wkv6_sequential(r, k, v, w, u=None):
    """RWKV-6 WKV 순차 스캔 (CPU/GPU 폴백)

    Args:
        r: (B, n_heads, T, headdim) — receptance
        k: (B, n_heads, T, headdim) — key
        v: (B, n_heads, T, headdim) — value
        w: (B, n_heads, T, headdim) — decay (log space, 음수)
        u: (n_heads, headdim) or None — in-context bonus (RWKV-6 Eagle)
            out[t] += (u * k[t]) · v[t] · r[t] per head

    Returns:
        (B, n_heads, T, headdim) — output
    """
    B, H, T, D = r.shape
    dtype = r.dtype

    # float32로 연산 (안정성)
    r, k, v, w = r.float(), k.float(), v.float(), w.float()
    if u is not None:
        u = u.float()  # (H, D)

    out = torch.zeros(B, H, T, D, device=r.device, dtype=torch.float32)
    # state[b,h,k,v] = h[k,v]: k ⊗ v 외적의 누적 상태
    # k-dim(첫번째)으로 합산하여 v-dim(두번째) 출력
    state = torch.zeros(B, H, D, D, device=r.device, dtype=torch.float32)

    if u is not None:
        u_f = u.float()  # (H, D=K)

    for t in range(T):
        rt = r[:, :, t, :]           # (B, H, D=K)
        kt = k[:, :, t, :]           # (B, H, D=K)
        vt = v[:, :, t, :]           # (B, H, D=V)
        wt = w[:, :, t, :]           # (B, H, D=K), 음수 log-space

        # kv = k ⊗ v : kv[b,h,k,v] = k[b,h,k] * v[b,h,v]
        kv = kt.unsqueeze(-1) * vt.unsqueeze(-2)  # (B, H, K, V)

        # output: fla 공식 — o[v] = sum_k (h[k,v] + u[k]*kv[k,v]) * r[k]
        # u[k]*kv[k,v] = u[k]*k[k]*v[v]
        if u is not None:
            # (state + u*kv) * r → sum over K
            out_t = ((state + u_f.unsqueeze(0).unsqueeze(-1) * kv)
                     * rt.unsqueeze(-1)).sum(-2)    # (B, H, V)
        else:
            out_t = (state * rt.unsqueeze(-1)).sum(-2)  # (B, H, V)
        out[:, :, t, :] = out_t

        # state 업데이트: h[k,v] = exp(w[k]) * h[k,v] + k[k]*v[v]
        decay = torch.exp(wt).unsqueeze(-1)  # (B, H, K, 1)
        state = state * decay + kv

    return out.to(dtype)


class RWKV6TimeMix(nn.Module):
    """RWKV-6 TimeMix 블록

    입력 x → (r, k, v, w, g) 프로젝션 + WKV 순차 스캔

    프로젝션:
    - r_proj, k_proj, v_proj: BitLinear (1.58bit)
    - w_base + w_lora: 데이터 의존 감쇄 (음수 log space)
    - g_proj: 게이트 (FP Linear)
    - o_proj: 출력 프로젝션 (BitLinear)
    """

    def __init__(self, d_model: int, n_heads: int, headdim: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.headdim = headdim
        assert d_model == n_heads * headdim

        # 메인 프로젝션 (BitLinear — 1.58bit 양자화)
        self.r_proj = BitLinear(d_model, d_model)
        self.k_proj = BitLinear(d_model, d_model)
        self.v_proj = BitLinear(d_model, d_model)
        self.o_proj = BitLinear(d_model, d_model)

        # 게이트 (FP — sigmoid 활성화이므로 정밀도 유지)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)

        # 데이터 의존 감쇄 (w)
        # w_base: 학습 가능한 기본 감쇄 (음수 → exp(w) < 1)
        self.w_base = nn.Parameter(torch.empty(n_heads * headdim))
        nn.init.uniform_(self.w_base, -5.0, -1.0)
        # w_lora: 입력 의존 감쇄 조절
        lora_rank = max(16, d_model // 16)
        self.w_lora_down = nn.Linear(d_model, lora_rank, bias=False)
        self.w_lora_up = nn.Linear(lora_rank, d_model, bias=False)

        # in-context bonus (RWKV-6 Eagle u 파라미터)
        # 출력: out[t] += (u * k[t]) * (v[t] · r[t]) per head
        # gradient가 T에 비례하여 폭발하므로 1/T 스케일링 hook 등록
        self.u = nn.Parameter(torch.zeros(n_heads, headdim))
        self.u.register_hook(lambda grad: grad * 0.01)

        # per-head output normalization
        self.output_norm = nn.LayerNorm(headdim)

    def _init_weights(self):
        """LoRA up을 zero-init → 초기 w = w_base"""
        nn.init.zeros_(self.w_lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        B, T, D = x.shape

        # 프로젝션
        r = self.r_proj(x).view(B, T, self.n_heads, self.headdim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.headdim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.headdim).transpose(1, 2)

        # 데이터 의존 감쇄
        w = self.w_base + self.w_lora_up(torch.tanh(self.w_lora_down(x)))
        w = -F.softplus(w)   # 음수 보장 → exp(w) ∈ (0, 1)
        w = w.view(B, T, self.n_heads, self.headdim).transpose(1, 2)

        # 게이트
        g = torch.sigmoid(self.g_proj(x))  # (B, T, D)

        # WKV 스캔
        if _FLA_AVAILABLE and x.is_cuda:
            # fla fused_recurrent: custom autograd → backward 안정성 확보
            # bf16 overflow 방지: float32로 변환 후 연산
            r_fla = r.transpose(1, 2).float()  # (B, T, H, D)
            k_fla = k.transpose(1, 2).float()
            v_fla = v.transpose(1, 2).float()
            w_fla = w.transpose(1, 2).float()
            u_f32 = self.u.float()
            o, _ = _fused_recurrent_rwkv6(r_fla, k_fla, v_fla, w_fla, u_f32,
                                           scale=1.0, output_final_state=False)
            out = o.to(r.dtype)  # (B, T, H, D)
        else:
            out = wkv6_sequential(r, k, v, w, self.u)  # (B, H, T, D)
            out = out.transpose(1, 2)  # (B, T, H, D)

        # per-head normalization
        out = self.output_norm(out)  # (B, T, H, D)

        # reshape + gate
        out = out.reshape(B, T, D)
        out = out * g

        # 출력 프로젝션
        out = self.o_proj(out)

        return out
