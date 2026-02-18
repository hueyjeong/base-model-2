"""토크나이저 공통 인터페이스 (Abstract Base Class)

모든 토크나이저 구현체는 이 인터페이스를 상속하여 사용한다.
모델 코드는 BaseTokenizer 인터페이스에만 의존하므로,
토크나이저를 자유롭게 교체할 수 있다.
"""
from abc import ABC, abstractmethod
from typing import List, Optional


class BaseTokenizer(ABC):
    """토크나이저 공통 인터페이스"""

    # --- 필수 프로퍼티 ---

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """전체 어휘 크기"""
        ...

    @property
    @abstractmethod
    def pad_id(self) -> int:
        """패딩 토큰 ID"""
        ...

    @property
    @abstractmethod
    def bos_id(self) -> int:
        """시작 토큰 ID"""
        ...

    @property
    @abstractmethod
    def eos_id(self) -> int:
        """종료 토큰 ID"""
        ...

    @property
    @abstractmethod
    def unk_id(self) -> int:
        """미등록 토큰 ID"""
        ...

    # --- 필수 메서드 ---

    @abstractmethod
    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """텍스트 → 토큰 ID 리스트

        Args:
            text: 입력 텍스트
            add_special: True면 BOS/EOS 토큰 추가
        Returns:
            토큰 ID 리스트
        """
        ...

    @abstractmethod
    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """토큰 ID 리스트 → 텍스트

        Args:
            ids: 토큰 ID 리스트
            skip_special: True면 special token 제거
        Returns:
            디코딩된 텍스트
        """
        ...

    # --- 기본 구현 제공 ---

    def encode_batch(self, texts: List[str], add_special: bool = True) -> List[List[int]]:
        """배치 인코딩 (기본: 순차 처리)"""
        return [self.encode(t, add_special) for t in texts]

    def decode_batch(self, ids_batch: List[List[int]], skip_special: bool = True) -> List[str]:
        """배치 디코딩 (기본: 순차 처리)"""
        return [self.decode(ids, skip_special) for ids in ids_batch]

    def __len__(self) -> int:
        return self.vocab_size
