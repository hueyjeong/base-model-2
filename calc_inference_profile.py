"""
BitMamba Seq2Seq 추론 리소스/속도 프로파일 계산기

모델: 256M 프리셋 (d_model=768)
인코더 7층, 디코더 12층, Mamba-2 SSD, BitLinear, LinearCrossAttention
"""

# ── 모델 설정 ──────────────────────────────────
d_model = 768
d_inner = 1536      # Mamba 내부 차원
d_ff = 1280         # FFN 중간 차원 (BitNetFFN)
d_state = 128       # SSM state 차원
d_conv = 4          # conv1d 커널 크기
headdim = 64        # Mamba head 차원
ngroups = 1         # Mamba 그룹 수
nheads_mamba = d_inner // headdim  # 24
n_heads_attn = d_model // 64       # 12 (cross-attention)
d_head_attn = 64
vocab_size = 303

n_enc = 7
n_dec = 12

# Mamba2 in_proj 출력 차원
proj_dim = 2 * d_inner + 2 * ngroups * d_state + nheads_mamba  # 3352
conv_channels = d_inner + 2 * ngroups * d_state  # 1792

print("=" * 60)
print("BitMamba Seq2Seq 추론 리소스 프로파일")
print("=" * 60)

# ── 1. 파라미터 수 ──────────────────────────────
print("\n[1] 파라미터 수")
print("-" * 60)

# 임베딩 (tied: encoder_emb = decoder_emb = lm_head)
p_embed = vocab_size * d_model
print(f"  임베딩 (tied×3):         {p_embed:>12,} ({p_embed/1e6:.2f}M)")

# 인코더 레이어
p_mamba2 = (
    d_model * proj_dim +          # in_proj
    conv_channels * d_conv +      # conv1d weight
    conv_channels +               # conv1d bias
    d_inner +                     # norm
    d_inner * d_model +           # out_proj
    nheads_mamba * 3              # dt_bias + A_log + D
)
p_enc_ffn = 3 * d_model * d_ff   # gate + up + down (BitLinear, weight only)
p_enc_norm = d_model * 2          # norm1 + norm2
p_enc_layer = p_mamba2 + p_enc_ffn + p_enc_norm
p_enc_total = p_enc_layer * n_enc

print(f"  인코더 레이어 (×{n_enc}):")
print(f"    Mamba2:                {p_mamba2:>12,}")
print(f"    BitNetFFN:             {p_enc_ffn:>12,}")
print(f"    RMSNorm ×2:            {p_enc_norm:>12,}")
print(f"    레이어 소계:           {p_enc_layer:>12,} ({p_enc_layer/1e6:.2f}M)")
print(f"    인코더 합계:           {p_enc_total:>12,} ({p_enc_total/1e6:.2f}M)")

# 디코더 레이어
p_cross_attn = 4 * d_model * d_model  # q,k,v,o proj (MHA, n_kv_heads=n_heads)
p_dec_ffn = 3 * d_model * d_ff
p_dec_norm = d_model * 3              # norm1 + norm_cross + norm2
p_dec_layer = p_mamba2 + p_cross_attn + p_dec_ffn + p_dec_norm
p_dec_total = p_dec_layer * n_dec

print(f"  디코더 레이어 (×{n_dec}):")
print(f"    Mamba2:                {p_mamba2:>12,}")
print(f"    CrossAttention:        {p_cross_attn:>12,}")
print(f"    BitNetFFN:             {p_dec_ffn:>12,}")
print(f"    RMSNorm ×3:            {p_dec_norm:>12,}")
print(f"    레이어 소계:           {p_dec_layer:>12,} ({p_dec_layer/1e6:.2f}M)")
print(f"    디코더 합계:           {p_dec_total:>12,} ({p_dec_total/1e6:.2f}M)")

# 기타
p_final_norm = d_model
p_copy_gate = d_model * 1 + 1  # weight + bias (gate → scalar)
p_other = p_final_norm + p_copy_gate

p_total = p_embed + p_enc_total + p_dec_total + p_other
print(f"  기타 (final_norm, copy_gate): {p_other:>8,}")
print(f"  ─────────────────────────────────────")
print(f"  총 파라미터 수:          {p_total:>12,} ({p_total/1e6:.1f}M)")

# ── 2. 메모리 사용량 ────────────────────────────
print(f"\n[2] 메모리 사용량")
print("-" * 60)

bytes_f32 = p_total * 4
bytes_f16 = p_total * 2
bytes_int8 = p_total * 1

# 실제 safetensors는 일부 텐서가 tied로 빠져있지만 로딩 시 clone됨
print(f"  모델 가중치 (F32):       {bytes_f32/1024**2:>8.1f} MB")
print(f"  모델 가중치 (F16):       {bytes_f16/1024**2:>8.1f} MB (가능)")
print(f"  모델 가중치 (INT8):      {bytes_int8/1024**2:>8.1f} MB (가능)")

# BitLinear pre-quant: ternary {-1,0,1} → 실제 INT2로 저장 가능
bitlinear_params = (p_enc_ffn * n_enc + p_dec_ffn * n_dec)
bytes_bitlinear_int2 = bitlinear_params * 2 / 8  # 2-bit
non_bitlinear_f32 = (p_total - bitlinear_params) * 4
print(f"  BitLinear INT2 + 나머지 F32: {(bytes_bitlinear_int2 + non_bitlinear_f32)/1024**2:>5.1f} MB (이론적)")

# 런타임 메모리: SSM state + cross-attn cache
ssm_state_per_layer = nheads_mamba * d_state * headdim * 4  # f32
conv_buf_per_layer = (d_conv - 1) * conv_channels * 4
mamba_state_total = (ssm_state_per_layer + conv_buf_per_layer) * n_dec

print(f"\n  런타임 상태 (디코더 incremental):")
print(f"    SSM state/layer:       {ssm_state_per_layer/1024:>8.1f} KB ({nheads_mamba}×{d_state}×{headdim})")
print(f"    Conv buf/layer:        {conv_buf_per_layer/1024:>8.1f} KB ({d_conv-1}×{conv_channels})")
print(f"    전체 Mamba state:      {mamba_state_total/1024:>8.1f} KB (×{n_dec} 레이어)")

# Cross-attn KV cache: (1, n_heads, d_head, d_head) + (1, n_heads, d_head)
kv_cache_per_layer = (n_heads_attn * d_head_attn * d_head_attn + n_heads_attn * d_head_attn) * 4
kv_cache_total = kv_cache_per_layer * n_dec
print(f"    CrossAttn KV cache:    {kv_cache_total/1024:>8.1f} KB (×{n_dec} 레이어)")
print(f"    런타임 상태 합계:      {(mamba_state_total + kv_cache_total)/1024:>8.1f} KB")

# 총 메모리
total_mem = bytes_f32 + mamba_state_total + kv_cache_total
print(f"\n  추론 총 메모리 (F32):    {total_mem/1024**2:>8.1f} MB")

# ── 3. FLOPs 분석 ───────────────────────────────
print(f"\n[3] FLOPs 분석 (1 MAC = 2 FLOPs)")
print("-" * 60)

# 인코더: L 토큰 전체 처리
def encoder_flops(L):
    per_layer = 0
    # Mamba2 forward
    per_layer += L * d_model * proj_dim * 2       # in_proj
    per_layer += L * conv_channels * d_conv * 2   # conv1d
    per_layer += L * nheads_mamba * d_state * headdim * 6  # SSM scan (state update + output)
    per_layer += L * d_inner * d_model * 2        # out_proj
    # FFN (BitLinear — 양자화 오버헤드 ~30% 추가)
    per_layer += L * d_model * d_ff * 2 * 1.3     # gate_proj
    per_layer += L * d_model * d_ff * 2 * 1.3     # up_proj
    per_layer += L * d_ff * d_model * 2 * 1.3     # down_proj
    # RMSNorm, activations 등 (무시할 수준)
    return per_layer * n_enc

# Cross-attn cache 초기화: L 토큰
def cross_attn_cache_flops(L):
    per_layer = 0
    per_layer += L * d_model * d_model * 2  # K proj
    per_layer += L * d_model * d_model * 2  # V proj
    per_layer += n_heads_attn * d_head_attn * L * d_head_attn * 2  # K^T @ V
    return per_layer * n_dec

# 디코더 1 스텝 (incremental)
def decoder_step_flops():
    per_layer = 0
    # Mamba2 step
    per_layer += d_model * proj_dim * 2         # in_proj
    per_layer += conv_channels * d_conv * 2     # conv1d (1 token)
    per_layer += nheads_mamba * d_state * headdim * 6  # SSM step
    per_layer += d_inner * d_model * 2          # out_proj
    # Cross-attention (cached)
    per_layer += d_model * d_model * 2          # Q proj
    per_layer += n_heads_attn * d_head_attn * d_head_attn * 2  # Q @ KV
    per_layer += d_model * d_model * 2          # O proj
    # FFN (BitLinear ×1.3 overhead)
    per_layer += d_model * d_ff * 2 * 1.3       # gate
    per_layer += d_model * d_ff * 2 * 1.3       # up
    per_layer += d_ff * d_model * 2 * 1.3       # down
    return per_layer * n_dec

# LM head + copy gate
def lm_head_flops():
    return d_model * vocab_size * 2 + d_model * 2  # matmul + gate

enc_f = encoder_flops(1)
dec_f = decoder_step_flops()
lm_f = lm_head_flops()

print(f"  인코더 FLOPs/토큰:      {enc_f/1e6:>8.1f} MFLOPs")
print(f"  KV 캐시 초기화/토큰:    {cross_attn_cache_flops(1)/1e6:>8.1f} MFLOPs")
print(f"  디코더 1-step FLOPs:    {dec_f/1e6:>8.1f} MFLOPs")
print(f"  LM Head FLOPs:          {lm_f/1e6:>8.1f} MFLOPs")
print(f"  디코더 total/step:      {(dec_f+lm_f)/1e6:>8.1f} MFLOPs")

# 다양한 입력 길이
print(f"\n  시나리오별 총 FLOPs:")
for src_len, tgt_len, label in [(23, 20, "짧은 (7자)"),
                                  (81, 80, "중간 (32자)"),
                                  (236, 120, "긴 (100자)")]:
    total = encoder_flops(src_len) + cross_attn_cache_flops(src_len) + \
            (decoder_step_flops() + lm_head_flops()) * tgt_len
    encode_part = encoder_flops(src_len) + cross_attn_cache_flops(src_len)
    decode_part = (decoder_step_flops() + lm_head_flops()) * tgt_len
    print(f"    {label:12s} (src={src_len:3d}, tgt={tgt_len:3d}): "
          f"{total/1e9:6.2f} GFLOPs  "
          f"(enc {encode_part/1e9:.2f}G + dec {decode_part/1e9:.2f}G)")

# ── 4. 속도 분석 (벤치마크 기반) ─────────────────
print(f"\n[4] 속도 분석 (i9-13900KS, single-thread)")
print("-" * 60)

# 벤치마크 결과 (모델 로드 시간 제외)
# 짧은: total 2.05s, user 1.58s → 모델로드~0.8s, 추론~0.78s
# 중간: total 4.05s, user 3.96s → 모델로드~0.8s, 추론~3.16s
# 긴:   total 6.20s, user 6.84s → 모델로드~0.8s, 추론~6.0s
benchmarks = [
    ("짧은 (7자)", 23, 20, 2.05, 0.8),
    ("중간 (32자)", 81, 80, 4.05, 0.8),
    ("긴 (100자)", 236, 120, 6.20, 0.8),
]

print(f"  {'시나리오':12s} {'총시간':>7s} {'추론':>7s} {'인코딩':>7s} {'디코딩':>7s} {'tok/s':>7s}")
for label, src, tgt, total_s, load_s in benchmarks:
    infer_s = total_s - load_s
    # 추정: 인코딩 비율 = src/(src+tgt), 실제는 인코더가 좀 더 무거움 (batch mode)
    enc_flops = encoder_flops(src) + cross_attn_cache_flops(src)
    dec_flops = (decoder_step_flops() + lm_head_flops()) * tgt
    total_flops = enc_flops + dec_flops
    enc_ratio = enc_flops / total_flops
    enc_s = infer_s * enc_ratio
    dec_s = infer_s * (1 - enc_ratio)
    tok_per_s = tgt / dec_s if dec_s > 0 else 0
    print(f"  {label:12s}  {total_s:5.2f}s  {infer_s:5.2f}s  {enc_s:5.2f}s  {dec_s:5.2f}s  {tok_per_s:5.1f}")

# 추정 디코딩 throughput
# 중간 케이스가 가장 정확 (로드 비율 작음)
infer_s = 4.05 - 0.8  # 3.25s
enc_f = encoder_flops(81) + cross_attn_cache_flops(81)
dec_f_total = (decoder_step_flops() + lm_head_flops()) * 80
total_f = enc_f + dec_f_total
dec_ratio = dec_f_total / total_f
dec_time = infer_s * dec_ratio
step_ms = dec_time / 80 * 1000

print(f"\n  추정 디코더 step 시간:   {step_ms:.1f} ms/step")
print(f"  추정 디코딩 throughput:  {80/dec_time:.1f} tok/s")

# GFLOPS achieved
gflops_achieved = total_f / 1e9 / infer_s
print(f"  추정 연산 처리량:        {gflops_achieved:.2f} GFLOPS")

# ── 5. 최적화 여지 ──────────────────────────────
print(f"\n[5] 추가 최적화 여지")
print("-" * 60)
print(f"  현재: single-thread scalar loops")
print(f"  1. 멀티스레딩 (rayon): matmul 병렬화 → ~4-8× 가능")
print(f"  2. SIMD (AVX2/AVX-512): BitLinear ternary matmul → ~4-16× 가능")
print(f"     ternary weights {-1,0,1}은 곱셈 없이 add/sub/skip으로 처리 가능")
print(f"  3. INT8 activation: candle INT8 matmul 활용 → ~2-4×")
print(f"  4. 임베딩 F16: 메모리 절약 (vocab 작아서 효과 미미)")
print(f"  5. Ternary weight packing (2-bit): 메모리 {bytes_f32/1024**2:.0f}MB → ~{bytes_bitlinear_int2/1024**2 + non_bitlinear_f32/1024**2:.0f}MB")

# ── 6. 배포 요구사항 ────────────────────────────
print(f"\n[6] 배포 최소 사양")
print("-" * 60)
print(f"  CPU:  아무 x86_64 (SSE4.2 이상 권장)")
print(f"  RAM:  {total_mem/1024**2:.0f} MB + OS overhead ≈ 1 GB")
print(f"  디스크: {bytes_f32/1024**2:.0f} MB (safetensors)")
print(f"  OS:   Linux/macOS/Windows (Rust cross-compile)")
print(f"  GPU:  불필요 (CPU only)")
print(f"  바이너리 크기: ~5 MB (static link)")

print(f"\n  권장 사양 (실시간 응답):")
print(f"  CPU:  AVX2 이상 (Intel 8th gen+ / AMD Zen2+)")
print(f"  RAM:  2 GB 이상")
print(f"  목표: <100ms latency → 멀티스레딩+SIMD 필수")
