"""
한자 전처리/후처리 모듈

전처리: 한자 → <|BOHJ|>NFD음가+번호<|EOHJ|>
후처리: <|BOHJ|>NFD음가+번호<|EOHJ|> → 한자
"""
import re
import unicodedata
from collections import defaultdict
from hanja.table import hanja_table

BOHJ = "<|BOHJ|>"
EOHJ = "<|EOHJ|>"

_POSTPROCESS_RE = re.compile(
    re.escape(BOHJ) + r"(.+?)\+(\d+)" + re.escape(EOHJ)
)


def build_tables():
    """hanja_table로부터 정방향/역방향 매핑 테이블 구축

    Returns:
        forward_map: {한자: (nfd_음가, 번호)}
        reverse_map: {(nfd_음가, 번호): 한자}
    """
    # 음가별로 한자를 코드포인트 순으로 모음
    reading_to_chars = defaultdict(list)
    for ch, reading in hanja_table.items():
        if len(ch) == 1:
            reading_to_chars[reading].append(ch)

    # 코드포인트 순 정렬 → 번호 부여
    for reading in reading_to_chars:
        reading_to_chars[reading].sort(key=lambda c: ord(c))

    forward_map = {}
    reverse_map = {}

    for reading, chars in reading_to_chars.items():
        nfd_reading = unicodedata.normalize("NFD", reading)
        for idx, ch in enumerate(chars):
            forward_map[ch] = (nfd_reading, idx)
            reverse_map[(nfd_reading, idx)] = ch

    return forward_map, reverse_map


# 모듈 로드 시 1회 구축
_forward_map, _reverse_map = build_tables()


def preprocess(text):
    """텍스트 내 한자를 <|BOHJ|>NFD음가+번호<|EOHJ|>로 치환"""
    result = []
    for ch in text:
        if ch in _forward_map:
            nfd_reading, idx = _forward_map[ch]
            result.append(f"{BOHJ}{nfd_reading}+{idx}{EOHJ}")
        else:
            result.append(ch)
    return "".join(result)


def postprocess(text):
    """<|BOHJ|>NFD음가+번호<|EOHJ|> 패턴을 한자로 복원"""
    def _replace(m):
        reading = m.group(1)
        # NFC로 합쳐졌을 수 있으므로 NFD로 변환해서 조회
        nfd_reading = unicodedata.normalize("NFD", reading)
        idx = int(m.group(2))
        ch = _reverse_map.get((nfd_reading, idx))
        if ch:
            return ch
        # 매칭 실패 시 그대로 유지
        return m.group(0)

    return _POSTPROCESS_RE.sub(_replace, text)


def get_stats():
    """테이블 통계"""
    reading_counts = defaultdict(int)
    for _, (reading, _) in _forward_map.items():
        nfc = unicodedata.normalize("NFC", reading)
        reading_counts[nfc] += 1
    return {
        "total_hanja": len(_forward_map),
        "unique_readings": len(reading_counts),
        "max_same_reading": max(reading_counts.values()),
        "max_reading_char": max(reading_counts, key=reading_counts.get),
    }


if __name__ == "__main__":
    stats = get_stats()
    print(f"총 한자: {stats['total_hanja']}자")
    print(f"고유 음가: {stats['unique_readings']}개")
    print(f"최다 동음 한자: {stats['max_reading_char']} ({stats['max_same_reading']}자)")

    print("\n=== 전처리 테스트 ===")
    tests = [
        "大韓民國",
        "漢字를 섞은 文章입니다.",
        "Hello, world!",
        "한글만 있는 문장",
        "国と國は同じ読み",
    ]
    for text in tests:
        pre = preprocess(text)
        post = postprocess(pre)
        match = text == post
        print(f"  원문: {text}")
        print(f"  전처리: {pre}")
        print(f"  복원: {post}")
        print(f"  일치: {match}")
        if not match:
            print("  !! 불일치 !!")
        print()
