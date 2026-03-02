"""
2단계 병렬 외래어 추출

Stage 1 (병렬):
  say N개 워커가 sample_ko.jsonl을 청크로 나눠 MeCab 파싱 →
  외래어 후보 단어를 /tmp/loanword_candidates.txt 에 스트리밍 저장

Stage 2 (단일):
  후보 파일 읽기 → 중복 제거 → 음운 규칙 적용 → JSON 저장

실행: PYTHONPATH=. .venv/bin/python build_foreign_dict.py [stage1|stage2|all]
"""
import json
import os
import re
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

CORPUS_PATH  = "corpus/sample_ko.jsonl"
CANDIDATES_PATH = "/tmp/loanword_candidates.txt"
OUT_PATH     = "error_generation/resources/foreign_words.json"
N_WORKERS    = cpu_count()
CHUNK_SIZE   = 2_000    # 워커당 줄 수 (메모리 절약)

# ── 외래어 heuristic ──────────────────────────────────────
HANGUL_RE = re.compile(r'^[가-힣]{2,12}$')
LOANWORD_ENDINGS = frozenset([
    "터", "더", "버", "너", "처", "저", "커", "거", "퍼", "허",
    "이", "리", "스", "즈", "트", "드", "프", "브", "크", "그",
    "케", "테", "레", "세", "메", "베", "페", "데", "네", "게",
    "카", "타", "라", "바", "파", "마", "나", "하",
    "클", "블", "플", "슬", "글",
    "인", "언", "온", "운", "안",
    "션", "존", "폰", "론", "번",
    "밍", "링", "킹", "팅", "딩", "핑", "싱", "빙",
    "업", "앱", "립", "킷", "닛", "핏",
])

def is_likely_loanword(word: str) -> bool:
    return bool(HANGUL_RE.match(word)) and word[-1] in LOANWORD_ENDINGS

# ── 음운 혼동 규칙 ─────────────────────────────────────────
PHONETIC_RULES: list[tuple[str, str]] = [
    ("에", "애"), ("애", "에"),
    ("워", "와"), ("와", "워"),
    ("위", "의"), ("의", "위"),
    ("어", "오"), ("오", "어"),
    ("어", "에"), ("에", "어"),
    ("야", "이아"), ("이아", "야"),
    ("여", "이어"), ("이어", "여"),
    ("요", "이오"), ("이오", "요"),
    ("가", "까"), ("까", "가"), ("고", "꼬"), ("꼬", "고"),
    ("구", "꾸"), ("꾸", "구"), ("기", "끼"), ("끼", "기"),
    ("다", "따"), ("따", "다"),
    ("바", "빠"), ("빠", "바"), ("보", "뽀"), ("뽀", "보"),
    ("사", "싸"), ("싸", "사"),
    ("세", "쎄"), ("쎄", "세"), ("소", "쏘"), ("쏘", "소"),
    ("수", "쑤"), ("쑤", "수"),
    ("자", "짜"), ("짜", "자"),
    ("라", "나"), ("나", "라"), ("로", "노"), ("노", "로"),
    ("루", "누"), ("누", "루"), ("리", "니"), ("니", "리"),
    ("레", "네"), ("네", "레"),
    ("터", "타"), ("타", "터"),
    ("더", "다"), ("다", "더"),
    ("버", "바"), ("바", "버"),
    ("쉬", "슈"), ("슈", "쉬"),
    ("케이", "케"), ("케", "케이"),
    ("테이", "테"), ("테", "테이"),
    ("레이", "레"), ("레", "레이"),
    ("세이", "세"), ("세", "세이"),
    ("버", "퍼"), ("퍼", "버"),
    ("브", "프"), ("프", "브"),
    ("파", "화"), ("화", "파"),
    ("피", "히"), ("히", "피"),
    ("멤버", "맴버"), ("맴버", "멤버"),
    ("센터", "센타"), ("센타", "센터"),
]
END_RULES: list[tuple[str, str]] = [
    ("터", "타"), ("타", "터"), ("드", "트"), ("트", "드"),
    ("크", "그"), ("그", "크"), ("프", "스"), ("스", "프"),
    ("브", "스"),
]

def apply_rules(word: str) -> list[str]:
    results: set[str] = set()
    for from_s, to_s in PHONETIC_RULES:
        idx = word.find(from_s)
        while idx != -1:
            new_w = word[:idx] + to_s + word[idx + len(from_s):]
            if new_w != word and len(new_w) >= 2:
                results.add(new_w)
            idx = word.find(from_s, idx + 1)
    for end_from, end_to in END_RULES:
        if word.endswith(end_from):
            results.add(word[:-len(end_from)] + end_to)
    if len(word) >= 2:
        if word[0] == '에':
            results.add('애' + word[1:])
        elif word[0] == '애':
            results.add('에' + word[1:])
    return [r for r in results if r != word and HANGUL_RE.match(r)]


# ── Stage 1: 워커 함수 ────────────────────────────────────
def extract_to_queue(args: tuple) -> int:
    """
    lines: list[str] 을 파싱해 후보 단어를 임시 파일에 append.
    각 워커가 자신의 임시 파일에 씀.
    반환: 추출된 단어 수
    """
    lines, worker_id = args
    import MeCab, mecab_ko_dic
    local_tagger = MeCab.Tagger(f"-d {mecab_ko_dic.DICDIR}")

    tmp_file = f"/tmp/loanwords_worker_{worker_id}.txt"
    count = 0
    with open(tmp_file, "a", encoding="utf-8") as fout:
        for line in lines:
            try:
                obj = json.loads(line)
                text = obj.get("text", "")
                if not text:
                    continue
                node = local_tagger.parseToNode(text)
                while node:
                    surface = node.surface.strip()
                    feat = node.feature
                    if (feat.startswith("NNG") or feat.startswith("NNP")) and is_likely_loanword(surface):
                        fout.write(surface + "\n")
                        count += 1
                    node = node.next
            except Exception:
                pass
    return count


def read_chunks(path: str, chunk_size: int):
    buf: list[str] = []
    worker_id = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            buf.append(line)
            if len(buf) >= chunk_size:
                yield (buf, worker_id % N_WORKERS)
                buf = []
                worker_id += 1
    if buf:
        yield (buf, worker_id % N_WORKERS)


def stage1():
    # 이전 임시 파일 정리
    for i in range(N_WORKERS):
        p = Path(f"/tmp/loanwords_worker_{i}.txt")
        if p.exists():
            p.unlink()

    t0 = time.time()
    print(f"[STAGE 1] 병렬 파싱 시작 ({N_WORKERS}코어, {CORPUS_PATH})")
    total_words = 0
    chunks_done = 0

    with Pool(N_WORKERS) as pool:
        for cnt in pool.imap_unordered(extract_to_queue, read_chunks(CORPUS_PATH, CHUNK_SIZE)):
            total_words += cnt
            chunks_done += 1
            if chunks_done % 500 == 0:
                print(f"  청크 {chunks_done:,} | 경과 {time.time()-t0:.0f}s | 누적 단어(중복포함): {total_words:,}")

    # 전체 워커 결과를 하나로 합치기
    print(f"\n[STAGE 1] 완료. 워커 파일 병합 중...")
    tmp_merged = CANDIDATES_PATH + ".tmp"
    worker_files = [Path(f"/tmp/loanwords_worker_{i}.txt") for i in range(N_WORKERS) if Path(f"/tmp/loanwords_worker_{i}.txt").exists()]
    with open(tmp_merged, "w", encoding="utf-8") as fout:
        for p in worker_files:
            fout.write(p.read_text(encoding="utf-8"))
    # 원자적으로 교체 (덮어씀 방지)
    import shutil
    shutil.move(tmp_merged, CANDIDATES_PATH)
    # 워커 파일 삭제
    for p in worker_files:
        if p.exists():
            p.unlink()

    sz = Path(CANDIDATES_PATH).stat().st_size
    print(f"[STAGE 1] 후보 파일: {CANDIDATES_PATH} ({sz//1_000_000:.0f} MB)")
    print(f"[STAGE 1] 총 경과: {time.time()-t0:.0f}s")


def stage2():
    t0 = time.time()
    cand_path = Path(CANDIDATES_PATH)
    if not cand_path.exists() or cand_path.stat().st_size == 0:
        print(f"[STAGE 2] 오류: 후보 파일이 없거나 비었습니다 ({CANDIDATES_PATH}). stage1을 먼저 실행하세요.")
        return
    print(f"[STAGE 2] 중복 제거 + 규칙 적용 중: {CANDIDATES_PATH} ({cand_path.stat().st_size//1_000_000} MB)")

    # 후보 중복 제거
    unique_words: set[str] = set()
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                unique_words.add(w)

    print(f"  후보 (중복제거): {len(unique_words):,}개")

    # 기존 사전 로드
    out = Path(OUT_PATH)
    result: dict[str, list[str]] = {}
    if out.exists() and out.stat().st_size > 2:
        try:
            with open(out, encoding="utf-8") as f:
                result = json.load(f)
        except Exception:
            result = {}

    print(f"  기존 사전: {len(result):,}개 단어")

    # 규칙 적용
    for word in unique_words:
        variants = apply_rules(word)
        if not variants:
            continue
        if word not in result:
            result[word] = []
        for v in variants:
            if v not in result[word]:
                result[word].append(v)
        for v in variants:
            if v not in result:
                result[v] = []
            if word not in result[v]:
                result[v].append(word)

    result = {k: list(set(v)) for k, v in result.items() if v and any(x != k for x in v)}

    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    total_pairs = sum(len(v) for v in result.values())
    print(f"[STAGE 2] 완료 ({time.time()-t0:.0f}s)")
    print(f"  총 단어 수: {len(result):,}")
    print(f"  총 오류 쌍 수: {total_pairs:,}")
    print(f"  저장: {out}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("stage1", "all"):
        stage1()
    if mode in ("stage2", "all"):
        stage2()
