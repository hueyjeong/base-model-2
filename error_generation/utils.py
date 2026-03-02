import os
import re
from typing import NamedTuple, List, Optional
import MeCab

class TokenInfo(NamedTuple):
    surface: str
    pos: str
    start: int
    end: int

# 전역 Tagger 캐시 (MeCab 인스턴스화 오버헤드 방지)
_TAGGER: Optional[MeCab.Tagger] = None

def get_tagger() -> MeCab.Tagger:
    global _TAGGER
    if _TAGGER is None:
        try:
            import mecab_ko_dic
            dicdir = mecab_ko_dic.DICDIR
            _TAGGER = MeCab.Tagger(f"-r /dev/null -d {dicdir}")
        except ImportError:
            # mecab-ko-dic이 없는 경우 시스템 기본값 사용 시도
            _TAGGER = MeCab.Tagger("-r /dev/null")
    return _TAGGER

def get_mecab_offsets(text: str) -> List[TokenInfo]:
    """
    텍스트를 MeCab으로 분석한 뒤, 원래 문장에서의 (start, end) 오프셋을 매핑하여 반환.
    주의: MeCab은 공백을 삼키므로, 원본 텍스트의 index를 추적해야 함.
    """
    tagger = get_tagger()
    parsed = tagger.parse(text).split('\n')
    
    tokens = []
    current_idx = 0
    
    for line in parsed:
        if line == 'EOS' or not line:
            break
        
        # MeCab 'pos' format: surface\tPOS
        if '\t' not in line:
            continue
            
        surface, feature = line.split('\t')
        pos = feature.split(',')[0] if ',' in feature else feature
        
        # 원본에서 surface가 시작되는 지점을 찾음 (공백 등을 무시하고 전진)
        start_idx = text.find(surface, current_idx)
        if start_idx == -1:
            # 형태소 치환(예: '해' -> '하', '아') 발생 엣지 케이스 처리
            # 복잡한 매핑을 피하기 위해 임시로 현재 인덱스 부여 (추후 고도화 필요)
            start_idx = current_idx
            end_idx = current_idx + len(surface)
        else:
            end_idx = start_idx + len(surface)
            current_idx = end_idx
            
        tokens.append(TokenInfo(surface=surface, pos=pos, start=start_idx, end=end_idx))
        
    return tokens

def replace_by_offset(text: str, start: int, end: int, replacement: str) -> str:
    """오프셋을 이용해 문자열을 안전하게 치환합니다."""
    return text[:start] + replacement + text[end:]

def swap_by_offset(text: str, start1: int, end1: int, start2: int, end2: int) -> str:
    """두 개의 떨어져 있는 텍스트 블록의 위치를 상호 교체합니다."""
    # start1이 start2보다 앞에 오도록 정렬
    if start1 > start2:
        start1, end1, start2, end2 = start2, end2, start1, end1
        
    part1_str = text[start1:end1]
    part2_str = text[start2:end2]
    
    # 텍스트 스왑 조립 (part1 자리에 part2, part2 자리에 part1)
    return text[:start1] + part2_str + text[end1:start2] + part1_str + text[end2:]
