"""
BOLT-Zone: Blur-aware Object Localization and Tracking for Baseball Strike Zone Judgment

실시간 CPU 기반 야구공 검출, 추적 및 스트라이크 판정 시스템

Modules:
    - detect: YOLO26n 기반 빠른 공 검출
    - refine: YOLO26n-OBB 기반 블러 스트릭 정밀 검출  
    - track: ByteTrack/BoT-SORT 기반 Multi-Object Tracking
    - gate: 가변연산 게이팅 로직
    - zone: 스트라이크존 판정 및 3D 궤적 분석
    - utils: 공통 유틸리티
"""

__version__ = "0.1.0"
__author__ = "YUjin"

from pathlib import Path

# 프로젝트 루트 경로
ROOT = Path(__file__).resolve().parent.parent
