# Phase 1.2: ELECTRA Two-head Untied Content Head 실험

## Context

현재 ELECTRA two-head 모델의 content head는 `h @ word_embeddings.T + bias` (tied embedding).
KoELECTRA의 임베딩은 RTD(판별)용으로 학습되어 교정 토큰 생성과 목적 불일치.
content_acc=86.8%가 병목 → untied head로 content 예측 전용 학습 공간을 확보하면 개선되는지 검증.

## 변경 범위

### 1. `electra_gec/model.py` — untied content head 옵션

`KoELECTRAGECToR.__init__`에 `tied_content: bool = True` 파라미터 추가:

```python
def __init__(self, model_name, dropout=0.1, single_head=False, tied_content=True):
    ...
    self.tied_content = tied_content
    if not single_head:
        self.action_head = nn.Linear(d, 4)
        if tied_content:
            self.content_bias = nn.Parameter(torch.zeros(V))
        else:
            # untied: 독립 Linear head (768 → 35000)
            self.content_head = nn.Linear(d, V)
            nn.init.xavier_uniform_(self.content_head.weight)
            nn.init.zeros_(self.content_head.bias)
```

`forward()`에서 분기:
```python
if self.tied_content:
    content_logits = F.linear(h, embed_w, self.content_bias)
else:
    content_logits = self.content_head(h)
```

나머지(predict, freeze/unfreeze 등)는 변경 없음 — content_logits 형태 동일.

**추가 파라미터**: 768 × 35,000 + 35,000 = ~27M params (~54MB fp16)

### 2. `electra_gec/train.py` — CLI 인자 추가

- `--no_tied_content` (store_false → `tied_content=False`)
- 모델 생성 시 전달: `KoELECTRAGECToR(..., tied_content=args.tied_content)`
- Loss/validate 로직 변경 없음 (content_logits 형태 동일)

### 3. `electra_gec/evaluate.py` — CLI 인자 추가

- `--no_tied_content` 추가
- 모델 생성 시 전달
- 평가 로직 변경 없음

### 4. 체크포인트 호환성

- untied 체크포인트에는 `content_head.weight`, `content_head.bias` 키가 존재
- tied 체크포인트에는 `content_bias` 키만 존재
- load_state_dict에서 자동 구분 (strict=True 기본)

## 핵심 파일

- `electra_gec/model.py` — content head 분기 (tied vs untied)
- `electra_gec/train.py` — `--no_tied_content` CLI 인자
- `electra_gec/evaluate.py` — `--no_tied_content` CLI 인자

## 학습 실행 예시

```bash
python -m electra_gec.train \
    --corpus corpus/sample_full.jsonl --text_key text \
    --val_corpus corpus/val_ko_50k_shuffled.jsonl \
    --no_tied_content \
    --stage1_steps 50000 --stage2_steps 200000 \
    --batch_size 32 --save_interval 25000 --val_every 500
```

## 검증

1. `python electra_gec/model.py` smoke test (tied_content=False forward/backward)
2. 학습 시작 → val cont_acc가 tied 대비 개선되는지 확인
3. 문자 레벨 P/R/F0.5 → tied(F0.5 ~0.70)와 비교
