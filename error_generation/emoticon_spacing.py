"""
이모티콘 띄어쓰기 오류 — 이모티콘 앞뒤 마침표/공백 처리.

NIKL PARA에서 ~6%를 차지하는 패턴:
  "네네. ㅋㅋ" → "네네ㅋㅋ"  (마침표+공백 제거)
  "괜찮아 ㅠㅠ" → "괜찮아ㅠㅠ"  (공백 제거)
  "ㅋㅋ." → "ㅋㅋ"  (이모티콘 뒤 마침표 제거)
"""

import random
import re
from typing import Optional

# 이모티콘 패턴 (자모 반복, 특수문자 이모티콘)
_EMOTICON_RE = re.compile(
    r'(?:'
    r'[ㅋㅎㅠㅜ]{2,}'          # ㅋㅋ, ㅎㅎ, ㅠㅠ, ㅜㅜ
    r'|[ㅋㅎㅠㅜ]'              # 단일 자모 (앞뒤 문맥으로 판단)
    r'|\^\^+'                   # ^^, ^^^
    r'|[><]{2,}'                # >>, <<, ><
    r'|ㅡㅡ+'                   # ㅡㅡ
    r'|[~]{2,}'                 # ~~
    r'|[.]{2,}'                 # .. (말줄임표도 이모티콘처럼 쓰임)
    r')'
)


def get_error_count() -> int:
    return 3  # 앞 마침표 제거, 앞 공백 제거, 뒤 마침표 제거


def apply_emoticon_spacing(text: str, rng: random.Random) -> Optional[str]:
    """이모티콘 앞뒤의 마침표/공백 제거 오류 주입."""
    emoticon_matches = list(_EMOTICON_RE.finditer(text))
    if not emoticon_matches:
        return None

    chosen = rng.choice(emoticon_matches)
    start, end = chosen.start(), chosen.end()

    actions = []

    # 이모티콘 앞에 ". " 또는 " " 가 있으면 제거 가능
    if start >= 2 and text[start - 2:start] == ". ":
        actions.append("remove_dot_space_before")
    if start >= 1 and text[start - 1] == " ":
        actions.append("remove_space_before")

    # 이모티콘 뒤에 "." 이 있으면 제거 가능
    if end < len(text) and text[end] in ".!?":
        actions.append("remove_punct_after")

    if not actions:
        return None

    action = rng.choice(actions)

    if action == "remove_dot_space_before":
        # ". ㅋㅋ" → "ㅋㅋ" 또는 " ㅋㅋ"
        if rng.random() < 0.5:
            return text[:start - 2] + text[start:]  # 마침표만 제거, 공백 유지
        else:
            return text[:start - 2] + text[start - 0:]  # 마침표+공백 제거 → 붙임

    elif action == "remove_space_before":
        # " ㅋㅋ" → "ㅋㅋ" (공백 제거)
        return text[:start - 1] + text[start:]

    elif action == "remove_punct_after":
        # "ㅋㅋ." → "ㅋㅋ"
        return text[:end] + text[end + 1:]

    return None
