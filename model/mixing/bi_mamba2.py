"""BiMamba2 Mixing Layer — 양방향 Mamba-2 SSD (Structured State Space Duality)

GPU 학습: nn.Linear proj + mamba_ssm fused CUDA kernel (mamba_split_conv1d_scan_combined)
CPU 추론: C chunk-parallel SSD 커널 (ctypes) — GPU와 수치 일치
         fallback: Python sequential scan (GPU와 ~10% 수치 차이 있음)

Mamba2Block: in_proj/out_proj는 nn.Linear — fused kernel 호환 + 최대 학습 속도.
Mamba2BitLinearBlock: in_proj 저랭크 + BitLinear, out_proj BitLinear — 전체 QAT 실험용.
"""
import ctypes
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.mixing.base import MixingLayer

# CUDA ops 감지
_MAMBA2_CUDA_OPS = False
_mamba_split_conv1d_scan_combined = None

try:
    import torch as _torch_check
    if _torch_check.cuda.is_available():
        from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined as _mscsc
        _mamba_split_conv1d_scan_combined = _mscsc
        _MAMBA2_CUDA_OPS = True
except ImportError:
    pass

# C chunk-parallel SSD 커널 (CPU fallback용 — GPU와 수치 일치)
_C_SSD_LIB = None
_c_mamba2_ssd_fwd = None

def _load_c_ssd_kernel():
    """inference_dense/c_kernels/libmamba2_ssd.so 로드. 없으면 빌드 시도."""
    global _C_SSD_LIB, _c_mamba2_ssd_fwd
    if _c_mamba2_ssd_fwd is not None:
        return True

    project_root = Path(__file__).resolve().parent.parent.parent
    so_path = project_root / "inference_dense" / "c_kernels" / "libmamba2_ssd.so"

    if not so_path.exists():
        # 자동 빌드 시도
        src = project_root / "inference_dense" / "c_kernels" / "mamba2_ssd.c"
        if src.exists():
            ret = os.system(
                f"gcc -O3 -march=native -mavx2 -mfma -fopenmp -shared -fPIC "
                f"-o {so_path} {src} -lm 2>/dev/null"
            )
            if ret != 0:
                return False

    if not so_path.exists():
        return False

    _C_SSD_LIB = ctypes.CDLL(str(so_path))
    _c_mamba2_ssd_fwd = _C_SSD_LIB.mamba2_ssd_fwd
    _c_mamba2_ssd_fwd.restype = None
    _c_mamba2_ssd_fwd.argtypes = [
        ctypes.c_void_p,  # x
        ctypes.c_void_p,  # B
        ctypes.c_void_p,  # C
        ctypes.c_void_p,  # dt
        ctypes.c_void_p,  # A
        ctypes.c_void_p,  # D
        ctypes.c_void_p,  # y (output)
        ctypes.c_int,     # chunk_size
        ctypes.c_int,     # seq_len
        ctypes.c_int,     # nheads
        ctypes.c_int,     # headdim
        ctypes.c_int,     # d_state
        ctypes.c_int,     # ngroups
    ]
    return True


def _ssd_scan_c(x_conv, b_conv, c_conv, dt, a_neg, d_skip, nheads, headdim, d_state, ngroups):
    """C chunk-parallel SSD forward — GPU fused kernel과 수치 일치

    Args:
        x_conv: (seq_len, d_inner) float32 contiguous — SiLU 적용됨
        b_conv: (seq_len, ngroups*d_state) float32
        c_conv: (seq_len, ngroups*d_state) float32
        dt: (seq_len, nheads) float32
        a_neg: (nheads,) float32 — -exp(A_log)
        d_skip: (nheads,) float32
    Returns:
        y: (seq_len, d_inner) float32 — scan output (D*x skip 포함)
    """
    seq_len = x_conv.shape[0]
    d_inner = nheads * headdim
    y = torch.zeros(seq_len, d_inner, dtype=torch.float32)

    _c_mamba2_ssd_fwd(
        x_conv.data_ptr(), b_conv.data_ptr(), c_conv.data_ptr(),
        dt.data_ptr(), a_neg.data_ptr(), d_skip.data_ptr(),
        y.data_ptr(),
        256,  # chunk_size
        seq_len, nheads, headdim, d_state, ngroups,
    )
    return y


class Mamba2Block(nn.Module):
    """Mamba-2 SSD 단방향 블록

    GPU: nn.Linear proj + mamba_ssm fused kernel (conv+scan+norm+gate+outproj)
    CPU: nn.Linear proj + Python sequential scan fallback

    파라미터 구조 (d=640, expand=2, d_state=64, headdim=64, ngroups=1):
      in_proj:  (2708, 640) — nn.Linear — z(1280) + xBC(1408) + dt(20)
      conv1d:   (1408, 1, 4) — depthwise on [x, B, C]
      norm:     (1280,) — RMSNorm weight (fused kernel 내부에서 사용)
      out_proj: (640, 1280) — nn.Linear
      A_log, D, dt_bias: (20,) each
    """

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4,
                 expand: int = 2, headdim: int = 64, ngroups: int = 1,
                 chunk_size: int = 256, int8_qat: bool = False):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.d_conv = d_conv
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        self.ngroups = ngroups
        self.chunk_size = chunk_size
        self.int8_qat = int8_qat

        assert self.d_inner % headdim == 0, \
            f"d_inner({self.d_inner})은 headdim({headdim})으로 나누어떨어져야 함"

        # 프로젝션 차원 (mamba_ssm 순서: z, xBC, dt)
        d_in_proj = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads
        d_conv_in = self.d_inner + 2 * ngroups * d_state

        if int8_qat:
            # INT8 QAT: 가중치+활성화 모두 INT8 양자화 (fused kernel 비호환 → fallback 사용)
            from model.bitlinear import Int8Linear
            self.in_proj = Int8Linear(d_model, d_in_proj, bias=False)
            self.out_proj = Int8Linear(self.d_inner, d_model, bias=False)
        else:
            # nn.Linear projection — fused kernel 호환, 최대 학습 속도
            self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)
            self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.conv1d = nn.Conv1d(
            d_conv_in, d_conv_in,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=d_conv_in, bias=True,
        )

        # RMSNorm weight — GPU에서는 fused kernel 내부에서 사용, CPU에서는 수동 적용
        self.norm = nn.RMSNorm(self.d_inner)

        # SSM 파라미터 (head당 스칼라)
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        A = torch.arange(1, self.nheads + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.nheads))

    def forward(self, x: Tensor, reset_mask: Tensor | None = None) -> Tensor:
        # int8_qat: fused kernel이 out_proj QAT를 우회하므로 fallback 강제
        if _MAMBA2_CUDA_OPS and x.is_cuda and not self.int8_qat:
            return self._forward_cuda(x, reset_mask)
        else:
            return self._forward_fallback(x, reset_mask)

    @torch.compiler.disable
    def _forward_cuda(self, x: Tensor, reset_mask: Tensor | None = None) -> Tensor:
        """nn.Linear in_proj → mamba_split_conv1d_scan_combined fused kernel"""
        # seq_idx for document isolation
        seq_idx = None
        if reset_mask is not None:
            seq_idx = (reset_mask.int().cumsum(dim=1) - 1).to(torch.int32)

        # 1) nn.Linear in_proj
        zxbcdt = self.in_proj(x)  # (B, T, d_in_proj)

        A = -torch.exp(self.A_log.float())

        # 2) Fused: conv1d + chunk_scan + RMSNorm + gate + out_proj
        y = _mamba_split_conv1d_scan_combined(
            zxbcdt,
            self.conv1d.weight.squeeze(1),  # (d_conv_in, d_conv)
            self.conv1d.bias,
            self.dt_bias,
            A,
            self.D,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            activation="silu",
            rmsnorm_weight=self.norm.weight,
            rmsnorm_eps=1e-5,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=False,
        )

        return y

    def _forward_fallback(self, x: Tensor, reset_mask: Tensor | None = None) -> Tensor:
        """CPU fallback — C chunk-parallel SSD 커널 사용 (GPU와 수치 일치)"""
        B_batch, T, _ = x.shape
        di = self.d_inner
        ds = self.d_state
        nh = self.nheads
        hd = self.headdim
        ng = self.ngroups
        d_conv_in = di + 2 * ng * ds

        # in_proj
        proj = self.in_proj(x)

        z = proj[:, :, :di]
        xBC_raw = proj[:, :, di:di + d_conv_in]
        dt_raw = proj[:, :, di + d_conv_in:]

        # conv1d on [x, B, C]
        xBC = self.conv1d(xBC_raw.transpose(1, 2))[:, :, :T].transpose(1, 2)

        # SiLU on ALL (GPU fused kernel activation="silu"와 동일)
        x_conv = F.silu(xBC[:, :, :di])
        B_conv = F.silu(xBC[:, :, di:di + ng * ds])
        C_conv = F.silu(xBC[:, :, di + ng * ds:])

        # dt
        dt = F.softplus(dt_raw + self.dt_bias)  # (B, T, nh)
        A_neg = -torch.exp(self.A_log.float())

        # SSD scan — C chunk-parallel 커널 (CPU) 또는 Python sequential (GPU/fallback)
        y_list = []
        use_c_kernel = not x.is_cuda and _load_c_ssd_kernel()
        for b in range(B_batch):
            x_b = x_conv[b].float().contiguous()       # (T, di)
            B_b = B_conv[b].float().contiguous()        # (T, ng*ds)
            C_b = C_conv[b].float().contiguous()        # (T, ng*ds)
            dt_b = dt[b].float().contiguous()            # (T, nh)
            A_b = A_neg.float().contiguous()             # (nh,)
            D_b = self.D.float().contiguous()            # (nh,)

            if use_c_kernel:
                y_b = _ssd_scan_c(x_b, B_b, C_b, dt_b, A_b, D_b, nh, hd, ds, ng)
            else:
                y_b = self._scan_sequential_single(
                    x_b, B_b, C_b, dt_b, A_b, D_b, nh, hd, ds, ng)
            y_list.append(y_b)

        # (B, T, di) — skip connection은 C 커널 내부에서 처리됨
        y = torch.stack(y_list, dim=0).to(x.dtype)

        # Reshape + gate→norm (norm_before_gate=False)
        y = y.reshape(B_batch, T, di)
        y = y * F.silu(z)
        y = self.norm(y)

        return self.out_proj(y)

    @staticmethod
    def _scan_sequential_single(x, B, C, dt, A_neg, D, nh, hd, ds, ng):
        """Sequential scan fallback (C 커널 없을 때) — 단일 시퀀스"""
        T = x.shape[0]
        heads_per_group = nh // ng
        decay = torch.exp(A_neg.unsqueeze(0) * dt)  # (T, nh)
        x_h = x.view(T, nh, hd)
        B_h = B.view(T, ng, ds)
        C_h = C.view(T, ng, ds)

        h = torch.zeros(nh, ds, hd, dtype=torch.float32, device=x.device)
        ys = []
        for t in range(T):
            a_t = decay[t].view(nh, 1, 1)
            b_t = B_h[t].repeat_interleave(heads_per_group, dim=0)
            c_t = C_h[t].repeat_interleave(heads_per_group, dim=0)
            h = a_t * h + b_t.unsqueeze(-1) * x_h[t].unsqueeze(1)
            y_t = (c_t.unsqueeze(-1) * h).sum(dim=1)
            ys.append(y_t)
        y = torch.stack(ys, dim=0)  # (T, nh, hd)
        y = y + D.view(1, nh, 1) * x_h
        return y.reshape(T, nh * hd)


class Mamba2BitLinearBlock(nn.Module):
    """Mamba-2 SSD 단방향 블록 — BitLinear QAT 버전

    in_proj: nn.Linear(d_model, rank) → BitLinear(rank, d_in_proj)  [저랭크]
    out_proj: BitLinear(d_inner, d_model)  [fused kernel에서 분리]
    SSM 파라미터(A_log, dt_bias, D) + state(h)는 FP32 유지.
    """

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4,
                 expand: int = 2, headdim: int = 64, ngroups: int = 1,
                 chunk_size: int = 256, in_proj_rank: int | None = None):
        super().__init__()
        from model.bitlinear import BitLinear

        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.d_conv = d_conv
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        self.ngroups = ngroups
        self.chunk_size = chunk_size

        assert self.d_inner % headdim == 0, \
            f"d_inner({self.d_inner})은 headdim({headdim})으로 나누어떨어져야 함"

        # 프로젝션 차원 (mamba_ssm 순서: z, xBC, dt)
        d_in_proj = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads
        d_conv_in = self.d_inner + 2 * ngroups * d_state

        # in_proj: 저랭크 분해 — nn.Linear(정밀 병목) → BitLinear(ternary 확장)
        rank = in_proj_rank or d_model
        self.in_proj_down = nn.Linear(d_model, rank, bias=False)
        self.in_proj_up = BitLinear(rank, d_in_proj)

        self.conv1d = nn.Conv1d(
            d_conv_in, d_conv_in,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=d_conv_in, bias=True,
        )

        # out_proj: BitLinear — fused kernel과 분리하여 별도 적용
        self.out_proj = BitLinear(self.d_inner, d_model)

        # RMSNorm weight — GPU에서는 fused kernel 내부, CPU에서는 수동 적용
        self.norm = nn.RMSNorm(self.d_inner)

        # SSM 파라미터 (FP32 유지 — head당 스칼라)
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        A = torch.arange(1, self.nheads + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.nheads))

    def forward(self, x: Tensor, reset_mask: Tensor | None = None) -> Tensor:
        if _MAMBA2_CUDA_OPS and x.is_cuda:
            return self._forward_cuda(x, reset_mask)
        else:
            return self._forward_fallback(x, reset_mask)

    @torch.compiler.disable
    def _forward_cuda(self, x: Tensor, reset_mask: Tensor | None = None) -> Tensor:
        """BitLinear in_proj → fused kernel (outproj 분리) → BitLinear out_proj"""
        seq_idx = None
        if reset_mask is not None:
            seq_idx = (reset_mask.int().cumsum(dim=1) - 1).to(torch.int32)

        # 1) 저랭크 BitLinear in_proj
        zxbcdt = self.in_proj_up(self.in_proj_down(x))

        A = -torch.exp(self.A_log.float())

        # 2) Fused: conv1d + chunk_scan + RMSNorm + gate (out_proj 제외)
        y = _mamba_split_conv1d_scan_combined(
            zxbcdt,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.dt_bias,
            A,
            self.D,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            activation="silu",
            rmsnorm_weight=self.norm.weight,
            rmsnorm_eps=1e-5,
            outproj_weight=None,  # out_proj 분리 — BitLinear 별도 적용
            outproj_bias=None,
            headdim=self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=False,
        )

        # 3) BitLinear out_proj
        return self.out_proj(y)

    def _forward_fallback(self, x: Tensor, reset_mask: Tensor | None = None) -> Tensor:
        """CPU fallback — C chunk-parallel SSD 커널 사용 (GPU와 수치 일치)"""
        B_batch, T, _ = x.shape
        di = self.d_inner
        ds = self.d_state
        nh = self.nheads
        hd = self.headdim
        ng = self.ngroups
        d_conv_in = di + 2 * ng * ds

        # 저랭크 BitLinear in_proj
        proj = self.in_proj_up(self.in_proj_down(x))

        z = proj[:, :, :di]
        xBC_raw = proj[:, :, di:di + d_conv_in]
        dt_raw = proj[:, :, di + d_conv_in:]

        # conv1d on [x, B, C]
        xBC = self.conv1d(xBC_raw.transpose(1, 2))[:, :, :T].transpose(1, 2)

        # SiLU on ALL (GPU fused kernel activation="silu"와 동일)
        x_conv = F.silu(xBC[:, :, :di])
        B_conv = F.silu(xBC[:, :, di:di + ng * ds])
        C_conv = F.silu(xBC[:, :, di + ng * ds:])

        # dt (FP32 강제)
        dt = F.softplus(dt_raw.float() + self.dt_bias.float())
        A_neg = -torch.exp(self.A_log.float())

        # C chunk-parallel SSD — batch=1씩 처리
        y_list = []
        for b in range(B_batch):
            x_b = x_conv[b].float().contiguous()
            B_b = B_conv[b].float().contiguous()
            C_b = C_conv[b].float().contiguous()
            dt_b = dt[b].float().contiguous()
            A_b = A_neg.float().contiguous()
            D_b = self.D.float().contiguous()

            if _load_c_ssd_kernel():
                y_b = _ssd_scan_c(x_b, B_b, C_b, dt_b, A_b, D_b, nh, hd, ds, ng)
            else:
                y_b = Mamba2Block._scan_sequential_single(
                    x_b, B_b, C_b, dt_b, A_b, D_b, nh, hd, ds, ng)
            y_list.append(y_b)

        y = torch.stack(y_list, dim=0).to(x.dtype)

        # Reshape + gate→norm (norm_before_gate=False)
        # y는 이미 (B, T, di), skip connection은 C 커널 내부에서 처리됨
        y = y * F.silu(z)
        y = self.norm(y)

        return self.out_proj(y)


class BiMamba2Mixing(MixingLayer):
    """양방향 Mamba-2 — forward + backward addition"""

    def __init__(self, cfg):
        super().__init__()
        ds = getattr(cfg, 'mamba2_d_state', 64)
        hd = getattr(cfg, 'mamba2_headdim', 64)
        ng = getattr(cfg, 'mamba2_ngroups', 1)
        cs = getattr(cfg, 'mamba2_chunk_size', 256)
        expand = getattr(cfg, 'mamba_expand', 2)
        d_conv = getattr(cfg, 'mamba_d_conv', 4)

        common_kwargs = dict(d_state=ds, d_conv=d_conv, expand=expand,
                             headdim=hd, ngroups=ng, chunk_size=cs)

        int8_qat = getattr(cfg, 'int8_qat', False)

        if getattr(cfg, 'bitlinear_mamba', False):
            rank = getattr(cfg, 'mamba2_in_proj_rank', None)
            self.fwd = Mamba2BitLinearBlock(cfg.d_model, **common_kwargs,
                                           in_proj_rank=rank)
            self.bwd = Mamba2BitLinearBlock(cfg.d_model, **common_kwargs,
                                           in_proj_rank=rank)
        else:
            self.fwd = Mamba2Block(cfg.d_model, **common_kwargs, int8_qat=int8_qat)
            self.bwd = Mamba2Block(cfg.d_model, **common_kwargs, int8_qat=int8_qat)

    def forward(self, x: Tensor, pad_mask: Tensor | None = None,
                reset_mask: Tensor | None = None) -> Tensor:
        fwd_out = self.fwd(x, reset_mask=reset_mask)
        if reset_mask is not None:
            bwd_reset = reset_mask.flip(1).clone()
            bwd_reset[:, 0] = True  # flipped 시퀀스 시작에 BOS 보장 (seq_idx >= 0)
        else:
            bwd_reset = None
        bwd_out = self.bwd(x.flip(1), reset_mask=bwd_reset).flip(1)
        out = fwd_out + bwd_out
        if pad_mask is not None:
            out = out * pad_mask.unsqueeze(-1).to(out.dtype)
        return out
