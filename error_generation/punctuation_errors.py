"""
구두점 오류 — 문장부호 누락, 오용, 반복, 정규화 역방향.

NIKL PARA 분석 결과 교정의 58%가 구두점 관련.
주요 패턴: 마침표 누락, 물음표 누락, 쉼표 삭제, 말줄임표 변형.
"""

import random
import re
from typing import Optional

_PUNCTS = [".", ",", "!", "?", "~"]


def get_error_count() -> int:
    """오류 패턴의 총 개수를 반환."""
    return 9  # 기존 3 + 신규 6


def apply_punctuation_error(text: str, rng: random.Random) -> Optional[str]:
    """구두점 오류 주입. 누락/반복/교체/정규화 역방향 등."""
    # 액션 가중치 (NIKL PARA 분포 반영: 누락 > 변형 > 삽입)
    actions = [
        ("remove_final",   30),  # 문장 끝 구두점 삭제 (가장 빈번)
        ("remove_comma",   15),  # 쉼표 삭제
        ("ellipsis_vary",  15),  # 말줄임표 변형
        ("duplicate",      10),  # 구두점 중복 (?→??, !→!!)
        ("swap_final",      8),  # 마침표↔물음표 교체
        ("delete_inner",    8),  # 기존: 내부 구두점 삭제
        ("repeat_inner",    7),  # 기존: 반복
        ("swap_inner",      4),  # 기존: 교체
        ("insert_random",   3),  # 기존: 무작위 삽입
    ]
    action_names = [a[0] for a in actions]
    action_weights = [a[1] for a in actions]

    # 최대 3번 시도 (선택된 액션이 적용 불가할 수 있음)
    for _ in range(3):
        [chosen] = rng.choices(action_names, weights=action_weights, k=1)

        result = _apply_action(text, chosen, rng)
        if result is not None and result != text:
            return result

    return None


def _apply_action(text: str, action: str, rng: random.Random) -> Optional[str]:
    """개별 구두점 액션 적용."""

    if action == "remove_final":
        # 문장 끝 구두점 삭제: "안녕하세요." → "안녕하세요"
        m = re.search(r'[.!?~…]+\s*$', text)
        if m:
            return text[:m.start()].rstrip()
        return None

    elif action == "remove_comma":
        # 쉼표 삭제: "아, 그래" → "아 그래"
        commas = [m for m in re.finditer(r',\s*', text)]
        if not commas:
            return None
        chosen = rng.choice(commas)
        # 쉼표 제거, 공백 하나로 치환
        return text[:chosen.start()] + " " + text[chosen.end():]

    elif action == "ellipsis_vary":
        # 말줄임표 변형: "..." ↔ "..", "......", "…" ↔ ".."
        # 먼저 유니코드 말줄임표(…) 체크
        if "…" in text:
            idx = text.index("…")
            variants = ["..", "...", "...."]
            return text[:idx] + rng.choice(variants) + text[idx + 1:]
        # ASCII 말줄임표
        m = re.search(r'\.{2,}', text)
        if m:
            dot_count = m.end() - m.start()
            # 원래 길이와 다른 길이로 변형
            candidates = [n for n in [1, 2, 4, 5, 6] if n != dot_count]
            if candidates:
                new_count = rng.choice(candidates)
                return text[:m.start()] + "." * new_count + text[m.end():]
        return None

    elif action == "duplicate":
        # 구두점 중복: "?" → "??", "!" → "!!"
        m = re.search(r'[?!.]\s*$', text)
        if m:
            punct = text[m.start()]
            count = rng.randint(2, 4)
            return text[:m.start()] + punct * count + text[m.end():]
        return None

    elif action == "swap_final":
        # 문장 끝 구두점 교체: "뭐 해?" → "뭐 해.", "좋아." → "좋아?"
        m = re.search(r'([.?!])\s*$', text)
        if not m:
            return None
        orig_punct = m.group(1)
        swaps = {"?": ".", ".": "?", "!": "."}
        new_punct = swaps.get(orig_punct, ".")
        return text[:m.start()] + new_punct

    elif action == "delete_inner":
        # 기존: 텍스트 내부의 구두점 삭제
        punct_matches = [(m.start(), m.end(), m.group())
                         for m in re.finditer(r"[.,!?~]+", text)]
        if not punct_matches:
            return None
        start, end, _ = rng.choice(punct_matches)
        return text[:start] + text[end:]

    elif action == "repeat_inner":
        # 기존: 구두점 반복
        punct_matches = [(m.start(), m.end(), m.group())
                         for m in re.finditer(r"[.,!?~]+", text)]
        if not punct_matches:
            return None
        start, end, punct_str = rng.choice(punct_matches)
        replacement = punct_str[0] * rng.randint(2, 4)
        return text[:start] + replacement + text[end:]

    elif action == "swap_inner":
        # 기존: 구두점 교체
        punct_matches = [(m.start(), m.end())
                         for m in re.finditer(r"[.,!?~]+", text)]
        if not punct_matches:
            return None
        start, end = rng.choice(punct_matches)
        replacement = rng.choice(_PUNCTS)
        return text[:start] + replacement + text[end:]

    elif action == "insert_random":
        # 기존: 무작위 위치에 구두점 삽입
        spaces = [m.start() for m in re.finditer(r"\s+", text)]
        if not spaces:
            if len(text) < 2:
                return None
            idx = rng.randint(1, len(text) - 1)
        else:
            idx = rng.choice(spaces)
        inserted = rng.choice(_PUNCTS)
        return text[:idx] + inserted + text[idx:]

    return None
