"""
BOLT-Track Module

ByteTrack/BoT-SORT 기반 Multi-Object Tracking 모듈
- Detect 결과에 ID 부여하여 시간축 일관성 유지
- 칼만 필터로 궤적 예측 및 보정
- Gate 모듈의 불확실성 신호 계산 지원
"""

__all__ = []
