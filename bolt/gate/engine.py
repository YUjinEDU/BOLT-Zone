"""
BOLT-Gate Engine

Adaptive Inference를 위한 게이팅 로직 구현.
Detect 단계의 불확실성(Uncertainty)을 기반으로 Refine(OBB) 수행 여부를 결정합니다.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

log = logging.getLogger(__name__)


class BoltGate:
    """
    BOLT-Zone Gating Engine
    
    Decision Flow:
    1. Detect Model Output (BBox, Confidence)
    2. Motion Analysis (Velocity, Acceleration) -> from Tracker
    3. Uncertainty Estimation (Blur Score, Size)
    4. Decision: Pass (0) vs Refine (1)
    """
    
    def __init__(
        self,
        conf_threshold: float = 0.6,    # 이보다 낮으면 Refine
        size_threshold: float = 30.0,   # 이보다 작으면(먼 거리) Refine
        blur_threshold: float = 0.5,    # (Future) 블러 추정치가 높으면 Refine
        enable_temporal: bool = True    # 궤적 기반 판단 사용 여부
    ):
        self.conf_th = conf_threshold
        self.size_th = size_threshold
        self.blur_th = blur_threshold
        self.enable_temporal = enable_temporal
        
        # 통계
        self.stats = {
            'total': 0,
            'refine': 0,
            'pass': 0
        }
    
    def decide(
        self,
        detection: Dict,
        track_info: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """
        Refine 수행 여부 결정
        
        Args:
            detection: {
                'bbox': [x1, y1, x2, y2],
                'conf': float,
                'class': int
            }
            track_info: {
                'velocity': [vx, vy],
                'cov_trace': float (칼만 필터 공분산/불확실성)
            }
            
        Returns:
            (should_refine: bool, reason: str)
        """
        self.stats['total'] += 1
        
        # 1. Confidence Check
        # 신뢰도가 낮으면 -> 블러로 인해 형태가 뭉개졌을 가능성 -> Refine
        if detection['conf'] < self.conf_th:
            self.stats['refine'] += 1
            return True, "Low Confidence"
            
        # 2. Size Check (Motion Blur Impact)
        # 물체가 작거나(멀리 있음) or 블러로 인해 bbox가 비정상적으로 길쭉함
        bbox = detection['bbox']
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        # 크기가 너무 작으면 OBB로 정밀하게 봐야 함
        if min(w, h) < self.size_th:
            self.stats['refine'] += 1
            return True, "Small Object"
            
        # 3. Temporal Uncertainty (if enabled)
        if self.enable_temporal and track_info:
            # 칼만 필터의 위치 불확실성이 크면 -> Refine으로 보정 필요
            if track_info.get('cov_trace', 0) > 50.0:  # 임계값 튜닝 필요
                self.stats['refine'] += 1
                return True, "High Trace Uncertainty"
                
            # 속도가 매우 빠르면 -> 모션 블러 심함 -> Refine
            speed = np.linalg.norm(track_info.get('velocity', [0, 0]))
            if speed > 30.0:  # px/frame (튜닝 필요)
                self.stats['refine'] += 1
                return True, "High Velocity"
        
        self.stats['pass'] += 1
        return False, "Clear View"
    
    def get_efficiency(self) -> float:
        """현재 Refine 비율 (낮을수록 연산 효율 좋음)"""
        if self.stats['total'] == 0:
            return 0.0
        return self.stats['refine'] / self.stats['total']
    
    def reset_stats(self):
        self.stats = {'total': 0, 'refine': 0, 'pass': 0}
