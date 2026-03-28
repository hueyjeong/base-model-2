"""
연속 문자 변형 — 구두점/이모티콘 반복 수 증감.

패턴:
  "..." → ".."    (축약)
  "..." → "......"  (확장)
  "ㅋㅋ" → "ㅋㅋㅋㅋㅋ"  (이모티콘 확장)
  "?" → "???"     (구두점 반복)
  "~" → "~~~"     (물결표 반복)
"""

import random
import re
from typing import Optional

# 반복 대상 문자 그룹
_REPEAT_RE = re.compile(
    r'(?:'
    r'[.]{2,}'       # 말줄임표
    r'|[?]{1,}'      # 물음표
    r'|[!]{1,}'      # 느낌표
    r'|[~]{1,}'      # 물결표
    r'|[ㅋ]{2,}'     # ㅋㅋ
    r'|[ㅎ]{2,}'     # ㅎㅎ
    r'|[ㅠ]{2,}'     # ㅠㅠ
    r'|[ㅜ]{2,}'     # ㅜㅜ
    r')'
)


def get_error_count() -> int:
    return 2  # 확장, 축약


def apply_char_repeat(text: str, rng: random.Random) -> Optional[str]:
    """연속 문자의 반복 수를 늘리거나 줄임."""
    matches = list(_REPEAT_RE.finditer(text))
    if not matches:
        return None

    chosen = rng.choice(matches)
    start, end = chosen.start(), chosen.end()
    matched = chosen.group()
    char = matched[0]
    cur_len = len(matched)

    # 확장 or 축약 선택
    if cur_len <= 1:
        # 단일 문자 → 확장만 가능
        new_len = rng.randint(2, 5)
    elif cur_len == 2:
        # 2개 → 확장(70%) or 축약(30%)
        if rng.random() < 0.3:
            new_len = 1
        else:
            new_len = rng.randint(3, 6)
    else:
        # 3개 이상 → 확장(40%) or 축약(60%)
        if rng.random() < 0.6:
            new_len = rng.randint(1, max(1, cur_len - 1))
        else:
            new_len = rng.randint(cur_len + 1, cur_len + 4)

    if new_len == cur_len:
        return None

    return text[:start] + char * new_len + text[end:]
