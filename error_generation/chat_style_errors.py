import random
import re
from typing import Optional

# 통신체 / 신조어 치환 패턴
_CHAT_STYLE_PATTERNS = [
    # ── 인사/응대 ──
    (re.compile(r"안녕하세요"), ["안냐세여", "안녕하세욤", "ㅎㅇ", "하이요"]),
    (re.compile(r"감사합니다"), ["감사요", "ㄱㅅㄱㅅ", "감쟈합니다", "고맙습니당", "감삼다"]),
    (re.compile(r"수고하셨습니다"), ["수고해씀다", "수고여", "수고수고", "ㅅㄱㅅㄱ"]),
    (re.compile(r"죄송합니다"), ["ㅈㅅ", "죄송요", "뎨송합니다", "ㅈㅅㅈㅅ"]),
    (re.compile(r"확인했습니다"), ["확인염", "ㅇㅋㅇㅋ", "확인여", "ㅇㅋ"]),
    (re.compile(r"괜찮아(요)?"), ["ㄱㅊ", "괜춘", "ㄱㅊㄱㅊ"]),

    # ── 구어 축약 (NIKL PARA 고빈도) ──
    (re.compile(r"뭐"), ["머", "뭬"]),
    (re.compile(r"싫어"), ["시러"]),
    (re.compile(r"맞아(요)?"), ["마자", "맞아용"]),
    (re.compile(r"그래(요)?"), ["그래용", "글쎄"]),
    (re.compile(r"아니(요|에요)?"), ["아뇨", "아녀", "아니용"]),
    (re.compile(r"하지 마"), ["하지마", "ㄴㄴ"]),
    (re.compile(r"모르겠어(요)?"), ["모르게써", "몰겄어", "몰루"]),
    (re.compile(r"알겠어(요)?"), ["알게써", "알겟어"]),
    (re.compile(r"어디"), ["어뒤", "어듸"]),
    (re.compile(r"왜요"), ["왜용", "와요"]),
    (re.compile(r"무엇"), ["뭣", "뭥미"]),
    (re.compile(r"그렇(지|죠)"), ["그치", "글치"]),
    (re.compile(r"어떡해"), ["어뜨케", "어떻해", "어케"]),

    # ── 초성어 (NIKL PARA 빈출) ──
    (re.compile(r"레알"), ["ㄹㅇ"]),
    (re.compile(r"인정"), ["ㅇㅈ", "ㅇㅈㅇㅈ"]),
    (re.compile(r"감사"), ["ㄱㅅ", "ㄱㅅㄱㅅ"]),
    (re.compile(r"응응"), ["ㅇㅇ"]),
    (re.compile(r"오키"), ["ㅇㅋ", "ㅇㅋㅇㅋ"]),
    (re.compile(r"맞아"), ["ㅁㅈ"]),
    (re.compile(r"고고"), ["ㄱㄱ"]),
    (re.compile(r"그니까"), ["ㄱㄴㄲ"]),
    (re.compile(r"괜찮아"), ["ㄱㅊ"]),

    # ── 구어 축약 (NIKL PARA 전수 분석 — 빈출순) ──
    (re.compile(r"나도"), ["나두"]),
    (re.compile(r"저도"), ["저두"]),
    (re.compile(r"그래도"), ["그래두"]),
    (re.compile(r"좀"), ["쫌"]),
    (re.compile(r"그냥"), ["걍"]),
    (re.compile(r"그리고"), ["글고", "글구", "그리구"]),
    (re.compile(r"많이"), ["마니"]),
    (re.compile(r"네가"), ["너가", "니가", "니"]),
    (re.compile(r"여하튼"), ["여튼"]),
    (re.compile(r"왜냐하면"), ["왜냐면"]),
    (re.compile(r"오래간만에"), ["간만에"]),
    (re.compile(r"먹고"), ["먹구"]),
    (re.compile(r"하고"), ["하구"]),

    # ── -라고요 → -라구요 (빈출 맞춤법) ──
    (re.compile(r"더라고요"), ["더라구요"]),
    (re.compile(r"으려고요"), ["으려구요"]),
    (re.compile(r"그렇죠"), ["그쵸", "그치"]),

    # ── 부사/감탄 ──
    (re.compile(r"너무"), ["넘", "넘무", "개", "존나"]),
    (re.compile(r"진짜"), ["찐", "레알", "렬루"]),
    (re.compile(r"엄청"), ["개", "존나", "완전"]),
    (re.compile(r"그러니까"), ["그니까", "긍까", "그니깐"]),
    (re.compile(r"나중에"), ["나중엔", "담에"]),

    # ── 유행어/밈 ──
    (re.compile(r"\b명작\b"), ["띵작"]),
    (re.compile(r"\b명언\b"), ["띵언"]),
    (re.compile(r"재미있어(요)?"), ["존잼", "꿀잼", "핵잼", "노잼"]),
    (re.compile(r"재미있게"), ["존잼으로", "재밌게"]),
    (re.compile(r"귀여워(요)?"), ["커여워", "졸귀", "귀욤"]),

    # ── 종결어미 변형 ──
    (re.compile(r"합니다"), ["함다", "합니당", "함니다"]),
    (re.compile(r"입니다"), ["임다", "입니당", "임니다"]),
    (re.compile(r"습니다"), ["슴다", "습니당", "슴니다"]),
    (re.compile(r"세요"), ["세용", "셈"]),
    (re.compile(r"어요"), ["어용"]),
    (re.compile(r"아요"), ["아용"]),
]

def get_error_count() -> int:
    """오류 패턴의 총 개수를 반환."""
    return len(_CHAT_STYLE_PATTERNS)

def apply_chat_style(text: str, rng: random.Random) -> Optional[str]:
    """통신체나 신조어를 주입."""
    matches = []
    for pattern, replacements in _CHAT_STYLE_PATTERNS:
        for m in pattern.finditer(text):
            matches.append((m, replacements))
    
    if not matches:
        return None
    
    match, replacements = rng.choice(matches)
    chosen_repl = rng.choice(replacements)
    
    return text[:match.start()] + chosen_repl + text[match.end():]
