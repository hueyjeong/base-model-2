# 토크나이저 비교 실험 가이드

## 목적

5종 토크나이저의 사전학습 효율을 **공정하게** 비교하여, GEC fine-tuning에 사용할 최적 토크나이저를 선별한다.

## 토크나이저 목록

| 이름 | Vocab | 단위 | 예시: "맞춤법을" |
|---|---|---|---|
| `bbpe` | 64,000 | 서브워드 | `[맞춤법][을]` |
| `mecab_bbpe` | 64,000 | 형태소+서브워드 | `[맞춤/법][을]` |
| `nfd` | ~300 | NFD 분해 유니코드 | `ㅁ/ㅏ/ㅈ/...` |
| `keyboard` | ~200 | 키보드 자모 | `ㅁ/ㅏ/ㅈ/...` |
| `char` | ~500 | 글자 | `맞/춤/법/을` |

---

## 왜 Cross-Entropy Loss로 직접 비교하면 불공정한가

같은 텍스트라도 토크나이저마다 **토큰 수**와 **vocab 크기**가 다르다:

```
"맞춤법을 확인해 주세요"

BBPE:     6 토큰  (vocab 64K 중 선택 → 어려운 분류)
Char:     12 토큰 (vocab 500 중 선택 → 쉬운 분류)
Keyboard: 30 토큰 (vocab 200 중 선택 → 더 쉬운 분류)
```

- vocab이 작을수록 per-token loss가 자연히 낮음
- 토큰 수가 다르면 같은 스텝에서 본 텍스트 양이 다름

→ **per-token loss 직접 비교 불가**

---

## 공정한 비교 지표: BPC (Bits Per Character)

### 공식

```
BPC = (loss × 총_토큰수) / (총_문자수 × ln(2))
```

### 의미

> "원본 텍스트 한 글자를 예측하는 데 평균 몇 비트가 필요한가"

토크나이저의 vocab 크기·토큰화 방식을 완전히 정규화하여, **동일한 척도**로 비교 가능.

### 예시

```
같은 텍스트 "맞춤법" (3글자):

BBPE: loss=11.0 × 1토큰 / (3글자 × 0.693) = 5.29 bpc
Char: loss=6.2  × 3토큰 / (3글자 × 0.693) = 8.94 bpc

→ BBPE가 같은 텍스트를 더 적은 비트로 표현 = 더 효율적
```

### 한계

BPC는 **언어 모델링(압축) 효율** 지표이다. GEC 능력을 직접 측정하지는 않는다.

- 사전학습 단계: BPC로 1차 스크리닝 (언어 이해력 proxy)
- 최종 판단: GEC fine-tuning 후 **F0.5 score**로 비교

---

## 공정한 학습량: `--max_chars`

같은 스텝이라도 토크나이저별로 처리하는 문자 수가 다르다:

```
pack_size=4096 기준, 1스텝에 처리하는 문자:
BBPE:     ~12,000자 (1토큰 ≈ 3자)
Char:      ~4,096자 (1토큰 ≈ 1자)
Keyboard:  ~2,000자 (1토큰 ≈ 0.5자)
```

따라서 **스텝이 아닌 총 문자 수**로 학습량을 통제해야 한다.

`--max_chars 500000000` → 모든 토크나이저가 정확히 **5억 문자**를 본 시점에서 종료.

---

## Chinchilla 최적 학습량

| 모델 크기 | 최적 토큰 수 (≈20×params) | 최적 문자 수 (BBPE 기준) |
|---|---|---|
| 8M | 160M 토큰 | **~500M 문자** |
| 32M | 640M 토큰 | ~2B 문자 |
| 64M | 1.28B 토큰 | ~4B 문자 |
| 128M | 2.56B 토큰 | ~8B 문자 |

---

## 실험 설정

### 8M 모델 토크나이저 비교 (1차 스크리닝)

**공통 설정:**

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u -m training.pretrain \
  --tokenizer {TOKENIZER} \
  --size 8M --corpus corpus/sample_1g.jsonl --text_key text \
  --pack_size 4096 --batch_size 4 --grad_accum_steps 1 \
  --grad_ckpt --amp --max_chars 500000000 --lr 5e-4 \
  --warmup_steps 1000 --log_every 10 --save_every 5000 \
  --val_corpus corpus/val_50k.jsonl --val_every 200 --val_steps 20 \
  --save_dir checkpoints/8M_{TOKENIZER}
```

`{TOKENIZER}`에 `bbpe`, `mecab_bbpe`, `nfd`, `keyboard`, `char` 대입.

### 비교 방법

1. 각 실험의 마지막 `val_bpc` 값을 비교 (낮을수록 좋음)
2. BPC 차이가 유의미하면 → 해당 토크나이저로 64M/128M 본학습
3. BPC 차이가 미미하면 → GEC 특성(형태소 경계, 오타 패턴)에 맞는 토크나이저 선택

### 예상 소요 시간

| GPU | 8M × 500M chars | 5종 전부 |
|---|---|---|
| RTX 5060 Ti 16GB | ~7시간 | ~35시간 |
| A100 80GB | ~2시간 | ~10시간 |
