"""
Blur-Enhanced Kalman Filter for Detection Smoothing

딥러닝 Detection 결과의 **노이즈(Flickering)**를 줄이고,
**탐지 실패(Miss) 시 블러 기반 속도로 더 정확한 예측**을 수행하는 칼만 필터.

핵심 기능:
- ✅ Constant Velocity (CV) 모델: 직선 궤적 가정
- ✅ Confidence-Adaptive R: 탐지 신뢰도 기반 노이즈 조정
- ✅ Miss count 기반 Track Lost 판정
- ✅ **블러 기반 속도 통합** (BlurBall 방식)

블러 속도 통합 (NEW):
    OBB에서 추출한 블러 정보(길이, 각도)를 활용하여 속도 상태를 보정합니다.
    Detection 성공 시 블러 속도와 Kalman 속도를 융합하여 더 정확한 추정.
    Detection Miss 시 마지막 블러 속도를 사용하여 예측 정확도 향상.

Reference:
    BlurBall (2025): blur_length = velocity * exposure_time
    MAE 84.4px → 53.0px (37% 개선)

⚠️ 중요 (LIMITATIONS):
    이 필터는 **장기 궤적을 "예측(Prediction)"하는 것이 아닙니다.**
    야구공의 실제 궤적은 Spin, Seam-Shifted Wake 등 측정 불가능한 변수에 의해
    최대 22cm까지 예측 불가능하게 변화할 수 있습니다.
    
    따라서 이 필터의 역할은:
    1. Detection Smoothing: BBox 떨림 감소
    2. Short-term Interpolation: 탐지 실패 시 2~3 프레임 동안 블러 기반 예측
    
    탐지가 3프레임 이상 연속으로 실패하면, Track Lost로 처리합니다.
"""

import numpy as np
from typing import Optional, Tuple
import logging

log = logging.getLogger(__name__)


class AdaptiveTracker:
    """Blur-Enhanced Kalman Filter with Confidence-Adaptive R
    
    CV 모델 기반 트래커 + 블러 속도 통합.
    OBB 블러 정보를 활용하여 속도 추정 정확도 향상.
    """
    
    MAX_MISS_FRAMES = 3  # 이 프레임 이상 탐지 실패 시 Track Lost
    
    def __init__(self, dt: float = 1/60.0, blur_velocity_weight: float = 0.3):
        """
        Args:
            dt: 프레임 간격 (초). 기본값 1/60초 (60fps)
            blur_velocity_weight: 블러 속도와 Kalman 속도 융합 시 블러 가중치 (0~1)
        """
        self.dt = dt
        self.blur_velocity_weight = blur_velocity_weight
        self.miss_count = 0
        self.is_lost = False
        
        # State: [px, py, pz, vx, vy, vz] (6-dim, Constant Velocity)
        self.x = np.zeros(6)
        
        # State Transition (F): x' = x + v*dt
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt
        
        # Measurement Matrix (H): Position만 관측
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        
        # Process Noise (Q)
        self.Q = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
        
        # Measurement Noise (R) - Adaptive하게 조정됨
        self.R_base = np.eye(3) * 0.1
        self.R = self.R_base.copy()
        
        # State Covariance (P)
        self.P = np.eye(6) * 10.0
        
        # Confidence penalty 계수
        self.alpha = 100.0
        
        # 블러 기반 속도 캐시 (Detection Miss 시 사용)
        self.last_blur_velocity = None
        self.blur_velocity_confidence = 0.0
    
    def predict(self) -> np.ndarray:
        """관성 예측 (블러 속도 활용)
        
        블러 속도가 캐시되어 있으면 Kalman 속도와 융합하여 예측.
        """
        # 블러 속도가 있고 신뢰도가 높으면 융합
        if self.last_blur_velocity is not None and self.blur_velocity_confidence > 0.5:
            # Kalman 속도와 블러 속도 융합
            kalman_vel = self.x[3:6].copy()
            blur_vel_3d = np.array([
                self.last_blur_velocity[0], 
                self.last_blur_velocity[1], 
                0.0  # Z축 블러는 2D에서 추정 불가
            ])
            
            # 가중 평균 (X, Y만 블러 적용)
            w = self.blur_velocity_weight
            fused_vel = kalman_vel.copy()
            fused_vel[0] = (1 - w) * kalman_vel[0] + w * blur_vel_3d[0]
            fused_vel[1] = (1 - w) * kalman_vel[1] + w * blur_vel_3d[1]
            
            # 융합된 속도로 상태 업데이트
            self.x[3:6] = fused_vel
        
        # 표준 Kalman 예측
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x[:3]
    
    def update(self, 
               measurement: Optional[np.ndarray] = None, 
               confidence: float = 0.0,
               blur_velocity: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Detection 결과로 상태 보정 (블러 속도 통합)
        
        Args:
            measurement: [x, y, z] or None if detection failed.
            confidence: 0.0 ~ 1.0 (YOLO confidence). 0 if no detection.
            blur_velocity: [vx, vy] 블러에서 추정한 2D 속도 (pixels/frame).
                          OBB에서 estimate_velocity_from_obb()로 계산.
        
        Returns:
            Corrected position [x, y, z] or None if track is lost.
        """
        # 블러 속도 캐시 업데이트
        if blur_velocity is not None and np.linalg.norm(blur_velocity) > 0.1:
            self.last_blur_velocity = blur_velocity.copy()
            self.blur_velocity_confidence = confidence
        
        if measurement is None or confidence < 0.1:
            # Detection Failed
            self.miss_count += 1
            if self.miss_count >= self.MAX_MISS_FRAMES:
                self.is_lost = True
                log.warning("Track Lost: Detection failed for 3+ frames.")
                return None
            # 블러 기반 예측 사용
            return self.predict()
        
        # Detection Success
        self.miss_count = 0
        self.is_lost = False
        
        # Confidence-Adaptive R: 신뢰도 낮으면 노이즈 증가
        penalty = 1.0 + self.alpha * ((1.0 - confidence) ** 2)
        self.R = self.R_base * penalty
        
        # Kalman Update (위치)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = measurement - (self.H @ self.x)
        self.x = self.x + (K @ y)
        self.P = (np.eye(6) - K @ self.H) @ self.P
        
        # 블러 속도로 Kalman 속도 보정 (Detection 성공 시)
        if blur_velocity is not None and np.linalg.norm(blur_velocity) > 0.1:
            self._fuse_blur_velocity(blur_velocity, confidence)
        
        return self.x[:3]
    
    def _fuse_blur_velocity(self, blur_velocity: np.ndarray, confidence: float):
        """블러 속도와 Kalman 속도 융합
        
        블러 속도는 순간 속도를 정확히 반영하므로,
        Kalman 속도와 가중 평균하여 더 정확한 속도 추정.
        
        Args:
            blur_velocity: [vx, vy] 블러 기반 2D 속도
            confidence: Detection confidence (높을수록 블러 신뢰)
        """
        kalman_vel = self.x[3:5]  # X, Y 속도만
        
        # 방향 일관성 체크 (180도 모호성 해결)
        dot_product = np.dot(kalman_vel, blur_velocity)
        if dot_product < 0:
            blur_velocity = -blur_velocity  # 방향 반전
        
        # 가중 평균 (confidence가 높을수록 블러 신뢰)
        w = self.blur_velocity_weight * confidence
        self.x[3] = (1 - w) * self.x[3] + w * blur_velocity[0]
        self.x[4] = (1 - w) * self.x[4] + w * blur_velocity[1]
        # Z축 속도는 변경하지 않음 (2D 블러에서 추정 불가)
    
    def get_velocity(self) -> np.ndarray:
        """현재 추정 속도 반환 (3D)"""
        return self.x[3:6]
    
    def get_velocity_2d(self) -> np.ndarray:
        """현재 추정 속도 반환 (2D, X-Y 평면)"""
        return self.x[3:5]
    
    def get_uncertainty(self) -> float:
        """위치 불확실성 (covariance trace)"""
        return np.trace(self.P[:3, :3])
    
    def get_blur_velocity(self) -> Optional[np.ndarray]:
        """마지막 블러 기반 속도 반환"""
        return self.last_blur_velocity
    
    def reset(self):
        """상태 초기화"""
        self.x = np.zeros(6)
        self.P = np.eye(6) * 10.0
        self.miss_count = 0
        self.is_lost = False
        self.last_blur_velocity = None
        self.blur_velocity_confidence = 0.0


if __name__ == "__main__":
    print("=== Blur-Enhanced Kalman Filter Test ===")
    
    tracker = AdaptiveTracker(dt=1/60.0, blur_velocity_weight=0.3)
    
    # 시뮬레이션: 직선 궤적 + 블러 속도
    print("\n[1] Detection + Blur Velocity")
    for i in range(10):
        measurement = np.array([100 + i*10, 200 + i*5, 0])
        blur_vel = np.array([10.0, 5.0])  # 블러에서 추정한 속도
        pos = tracker.update(measurement, confidence=0.9, blur_velocity=blur_vel)
        vel = tracker.get_velocity()
        print(f"Frame {i}: pos={pos[:2]}, vel={vel[:2]}")
    
    # Miss 테스트 (블러 속도 활용)
    print("\n[2] Miss Handling with Blur Velocity")
    for i in range(4):
        pos = tracker.update(None, confidence=0.0)
        print(f"Miss {i+1}: pos={pos}, blur_vel={tracker.get_blur_velocity()}, lost={tracker.is_lost}")
    
    # 초기화 테스트
    print("\n[3] Reset")
    tracker.reset()
    print(f"After reset: x={tracker.x}, blur_vel={tracker.get_blur_velocity()}")
