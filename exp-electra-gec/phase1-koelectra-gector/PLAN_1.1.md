# Phase 1.1: ELECTRA Single-head (70K tags) 실험

## Context

ELECTRA two-head(action+content 분리) 모델의 문자 레벨 F0.5가 ~0.70에서 포화.
DenseEditor single-head(608 tags)는 F0.5=0.833으로 대폭 우위.
**head 분리 vs vocab 크기** 중 어느 것이 결정적인지 분리 검증하기 위해
ELECTRA에 DenseEditor와 동일한 single-head 태그 체계(n_tags = 2+2V = 70,002)를 적용.

## 변경 범위

### 1. `electra_gec/model.py` — Single-head 모델 추가

기존 `KoELECTRAGECToR`를 수정하지 않고, **`--single_head` 플래그**로 분기:

```python
class KoELECTRAGECToR(nn.Module):
    def __init__(self, model_name, dropout=0.1, single_head=False):
        ...
        self.single_head = single_head
        V = self.vocab_size  # 35000
        if single_head:
            self.n_tags = 2 + 2 * V  # 70002
            self.tag_head = nn.Linear(d, self.n_tags)
        else:
            # 기존 two-head (action_head + content_bias)
            ...
```

- `forward()`: single_head면 `(tag_logits,)` 반환, two-head면 `(act_logits, cont_logits)` 반환
- `predict()`: single_head면 `tag_logits.argmax` → `tag_to_op()`로 action/content 분리 후 동일 인터페이스 반환
- keep_bias: single_head에서는 `tag_logits[:, :, TAG_KEEP] += keep_bias`
- conf_threshold: `softmax(tag_logits).max(dim=-1)` — 단일 확률로 자연스럽게 적용
- **메모리**: tag_head = 768 × 70,002 = ~54M params (~108MB fp16). 주의 필요하나 학습 가능

### 2. `electra_gec/dataset.py` — single_head 모드

- `__init__`에 `single_head: bool = False` 파라미터 추가
- `_tokenize_pair()`:
  - single_head=True: `compute_edit_tags()` 결과를 **그대로** 사용 (변환 없음)
  - single_head=False: 기존처럼 `single_to_two_head()` 변환
- yield 형태:
  - single_head: `{"input_ids", "attention_mask", "edit_tags"}` (기존 DenseEditor와 동일 형태)
  - two_head: 기존 `{"input_ids", "attention_mask", "action_tags", "content_tags"}`

### 3. `electra_gec/train.py` — single_head 학습 분기

- `--single_head` CLI 인자 추가
- 모델 생성: `KoELECTRAGECToR(..., single_head=args.single_head)`
- Loss:
  - single_head: `CrossEntropyLoss(weight=edit_weight, ignore_index=-100, label_smoothing=0.1)`
    - edit_weight: TAG_KEEP=1.0, 나머지=edit_loss_weight (DenseEditor와 동일)
  - two_head: 기존 act_criterion + cont_criterion 유지
- `validate()`: single_head면 단일 logits로 edit_p/r 계산
- 데이터셋: `WordPieceGECDataset(..., single_head=args.single_head)`
- `collate_dynamic_pad()`: single_head면 `edit_tags` 키 사용

### 4. `electra_gec/evaluate.py` — single_head 평가

- `--single_head` CLI 인자 추가
- `correct_text()`: single_head면 `model.predict()` → `apply_edit_tags()` (from model.edit_tags) 사용
- 나머지 char_edits 비교 로직은 동일

## 핵심 파일

- `electra_gec/model.py` — 모델 (single_head 분기 추가)
- `electra_gec/dataset.py` — 데이터셋 (태그 변환 분기)
- `electra_gec/train.py` — 학습 (loss/validate 분기)
- `electra_gec/evaluate.py` — 평가 (predict 분기)
- `model/edit_tags.py` — `apply_edit_tags()`, `tag_to_op()` 재사용

## 학습 실행 예시

```bash
# Single-head, Stage 1만 (heads_only) 빠른 실험
python -m electra_gec.train \
    --corpus corpus/sample_full.jsonl --text_key text \
    --val_corpus corpus/val_ko_50k_shuffled.jsonl \
    --single_head \
    --stage1_steps 50000 --stage2_steps 200000 \
    --batch_size 16 --max_seq_len 256 \
    --save_interval 25000 --val_every 500
```

주의: batch_size를 줄여야 할 수 있음 (tag_head 54M params 추가로 GPU 메모리 증가)

## 평가 실행 예시

```bash
python -m electra_gec.evaluate \
    --checkpoint electra_gec/checkpoints/step_50000.pt \
    --corpus corpus/val_ko_50k_shuffled.jsonl --text_key text \
    --single_head --n_samples 500
```

## 검증

1. `python electra_gec/model.py` smoke test (single_head=True forward/backward)
2. `python electra_gec/dataset.py -f corpus/val_ko_50k_shuffled.jsonl -k text --single_head` 데이터셋 확인
3. 단일 GPU 학습 시작 후 loss 감소 확인
4. 체크포인트 로드 → evaluate.py로 문자 레벨 P/R/F0.5 측정
5. two-head 결과(F0.5 ~0.70)와 비교
