"""
BOLT-Gate Module

가변연산 게이팅 로직 모듈
- 상태 머신: Idle, Acquire, Track, End
- 불확실성 신호 평가: low_conf, high_resid, high_jitter
- Track 상태에서만 Refine ON/OFF 결정
- CPU 자원 절약의 핵심 모듈
"""

__all__ = []
