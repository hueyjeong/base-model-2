# corpus_v3 — jamo-codec 전용 v3 수준 한국어 코퍼스 빌더

KoELECTRA-Small과의 아키텍처 비교를 위한 ~34GB 한국어 사전학습 코퍼스 구축 파이프라인.

## 구성

| 범주 | 소스 | 목표 크기 | 빌더 |
|---|---|---|---|
| 위키 | `wikimedia/wikipedia` (20231101.ko) | 전체 (~1.3GB) | `build_hf.py` |
| 나무위키 | `heegyu/namuwiki-extracted` | 전체 (~6GB) | `build_hf.py` |
| 웹 | `HuggingFaceFW/fineweb-2` (kor_Hang) | 5GB 스트리밍 | `build_hf.py` |
| NIKL 신문 | `NIKL_NEWSPAPER_v2 + 2020~2024` | 12GB 샘플링 | `build_nikl.py` |
| NIKL 문어 | `nikl_written + np + raw` | 전체 (~4.2GB) | `build_nikl.py` |
| NIKL 구어 | `nikl_spoken + dialogue + om` | 전체 (~4.0GB) | `build_nikl.py` |
| NIKL 메신저 | `nikl_messenger` | 전체 (~0.08GB) | `build_nikl.py` |
| **합계 (raw 기준)** | | **~32GB** | |

## 정제 규칙 (`corpus/`의 기존 파이프라인과 동일)

- `language_score >= 0.8` (fineweb-2 해당 시)
- `len(text) >= 30자` 필터
- `> 1000자`: 문장 단위(`.?!。` + 공백/줄바꿈)로 1k 버퍼 패킹
- NIKL: 문서 ID(`WARW...` 앞부분) 기준 그루핑 → 1000자 청크로 재패킹
- HTML 태그 strip (nikl_newspaper_2024 등)
- 최종 MD5 앞 8바이트 기반 정확 중복 제거

**KAGAS 평가셋 중복 제거는 적용하지 않음** (jamo-codec 평가는 KAGAS 미사용).

## 파이프라인 (3단계)

```
HF 스트리밍 + NIKL JSONL
   ↓  build_hf.py / build_nikl.py
/tmp/jamo_v3/clean/
   ├── wiki_ko.parquet
   ├── namuwiki.parquet
   ├── fineweb2_ko.parquet
   ├── nikl_newspaper.parquet
   ├── nikl_written.parquet
   ├── nikl_spoken.parquet
   └── nikl_messenger.parquet
   ↓  merge_split.py (MD5 dedup + 98/1/1 split)
/tmp/jamo_v3/final/*.parquet (SSD에서 임시 처리)
   ↓  shutil.move
corpus/jamo-codec-v3/
   ├── train.parquet  (~98%)
   ├── val.parquet    (~1%)
   └── test.parquet   (~1%)
```

출력 포맷: `parquet + zstd + text: string` (기존 `corpus/train.parquet`, `editor_dataset._iter_parquet`와 호환).

## 사용법

### 1. HF 소스 정제

```bash
cd /workspace/base-model-2
source .venv/bin/activate

# 전체
python exp-jamo-codec/data/corpus_v3/build_hf.py --sources all

# 개별
python exp-jamo-codec/data/corpus_v3/build_hf.py --sources wiki_ko
python exp-jamo-codec/data/corpus_v3/build_hf.py --sources namuwiki
python exp-jamo-codec/data/corpus_v3/build_hf.py --sources fineweb2_ko --target_gb 5
```

### 2. NIKL 정제

```bash
# 전체
python exp-jamo-codec/data/corpus_v3/build_nikl.py --sources all

# 신문 샘플링 크기 조정
python exp-jamo-codec/data/corpus_v3/build_nikl.py \
    --sources all --newspaper_target_gb 12
```

### 3. 합치기 + split

```bash
# 기본 (text 컬럼만 유지)
python exp-jamo-codec/data/corpus_v3/merge_split.py

# source 컬럼도 유지 (도메인별 혼합비 분석용)
python exp-jamo-codec/data/corpus_v3/merge_split.py --keep_source
```

## 주의사항

- 원본 HF 다운로드 캐시는 `~/.cache/huggingface/datasets/`에 쌓임 (수 GB)
- `/tmp/jamo_v3/clean/` 중간 산출물은 정리 전까지 남음 (~15GB)
- 최종 `corpus/jamo-codec-v3/`는 HDD (`/mnt/d/`) 위치
- 라이선스
  - Wikipedia: CC-BY-SA
  - NamuWiki: CC-BY-NC-SA 2.0 (비상업 주의)
  - FineWeb-2: ODC-By
  - NIKL (모두의 말뭉치): 연구용
