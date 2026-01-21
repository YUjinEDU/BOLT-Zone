"""
Temporal Feature Buffer for Multi-Frame Input

과거 프레임의 Detection 결과를 버퍼에 저장하고,
GateNet 및 Tracker에 Temporal Context를 제공합니다.

설계 원칙:
- YOLO 모델 자체는 변경 없이 (싱글 프레임 입력 유지)
- GateNet의 피처를 확장하여 temporal 정보 활용
- 실시간 스트리밍 호환 (미래 프레임 사용 안 함)

Reference:
- BlurBall (2025): 3-step temporal input
- Temporal Shift Module (TSM, ICCV 2019)
"""

import numpy as np
from collections import deque
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

log = logging.getLogger(__name__)


@dataclass
class FrameDetection:
    """단일 프레임의 Detection 결과"""
    frame_id: int
    timestamp: float  # 초 단위
    center: np.ndarray  # (x, y) 픽셀 좌표
    bbox: np.ndarray  # (x1, y1, x2, y2) 또는 (cx, cy, w, h)
    confidence: float
    obb: Optional[np.ndarray] = None  # (cx, cy, w, h, angle)
    velocity: Optional[np.ndarray] = None  # (vx, vy) 추정 속도


@dataclass
class TemporalContext:
    """Temporal 피처 (GateNet 입력용)"""
    # 속도 변화량 (가속도)
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(2))
    
    # 궤적 일관성 점수 (0~1, 직선에 가까울수록 1)
    trajectory_consistency: float = 0.5
    
    # 연속 탐지 프레임 수
    consecutive_detections: int = 0
    
    # 평균 confidence (최근 N 프레임)
    avg_confidence: float = 0.0
    
    # 속도 분산 (움직임 안정성)
    velocity_variance: float = 0.0
    
    # 예상 다음 위치 (Tracker 예측과 비교용)
    predicted_next_position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    
    def to_feature_vector(self) -> np.ndarray:
        """GateNet 입력용 피처 벡터 변환 (6-dim)"""
        return np.array([
            self.acceleration[0],           # 0: ax (normalized)
            self.acceleration[1],           # 1: ay (normalized)
            self.trajectory_consistency,    # 2: 궤적 일관성
            min(self.consecutive_detections / 10.0, 1.0),  # 3: 연속 탐지 (normalized)
            self.avg_confidence,            # 4: 평균 confidence
            min(self.velocity_variance, 1.0)  # 5: 속도 분산 (clamped)
        ], dtype=np.float32)


class TemporalFeatureBuffer:
    """시간적 특징 버퍼
    
    최근 N개 프레임의 Detection 결과를 저장하고,
    Temporal Context를 계산하여 GateNet에 제공합니다.
    
    Args:
        buffer_size: 저장할 최대 프레임 수 (default: 5)
        fps: 비디오 프레임 레이트 (속도 계산용)
    
    Example:
        >>> buffer = TemporalFeatureBuffer(buffer_size=5, fps=60)
        >>> buffer.add_detection(frame_id=0, center=[100, 200], confidence=0.9)
        >>> buffer.add_detection(frame_id=1, center=[110, 205], confidence=0.85)
        >>> context = buffer.get_temporal_context()
        >>> print(context.trajectory_consistency)
    """
    
    def __init__(self, buffer_size: int = 5, fps: float = 60.0):
        self.buffer_size = buffer_size
        self.fps = fps
        self.dt = 1.0 / fps
        
        self.buffer: deque[FrameDetection] = deque(maxlen=buffer_size)
        self.miss_count = 0
        
    def add_detection(
        self,
        frame_id: int,
        center: np.ndarray,
        confidence: float,
        bbox: Optional[np.ndarray] = None,
        obb: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None
    ):
        """새 Detection 결과 추가"""
        if timestamp is None:
            timestamp = frame_id * self.dt
        
        # 속도 계산 (이전 프레임과 비교)
        velocity = None
        if len(self.buffer) > 0:
            prev = self.buffer[-1]
            dt = timestamp - prev.timestamp
            if dt > 0:
                velocity = (np.array(center) - prev.center) / dt
        
        detection = FrameDetection(
            frame_id=frame_id,
            timestamp=timestamp,
            center=np.array(center),
            bbox=np.array(bbox) if bbox is not None else np.zeros(4),
            confidence=confidence,
            obb=np.array(obb) if obb is not None else None,
            velocity=velocity
        )
        
        self.buffer.append(detection)
        self.miss_count = 0
        
    def add_miss(self, frame_id: int, timestamp: Optional[float] = None):
        """탐지 실패 기록"""
        self.miss_count += 1
        # Miss 시에는 버퍼에 추가하지 않음 (마지막 유효 Detection 유지)
        
    def get_temporal_context(self, image_size: Tuple[int, int] = (1280, 720)) -> TemporalContext:
        """Temporal Context 계산
        
        Args:
            image_size: (width, height) 정규화용
            
        Returns:
            TemporalContext: GateNet 입력용 temporal 피처
        """
        img_w, img_h = image_size
        context = TemporalContext()
        
        if len(self.buffer) < 2:
            return context
        
        # 최근 Detection들
        detections = list(self.buffer)
        n = len(detections)
        
        # 1. 가속도 계산 (최근 3개 프레임 사용)
        if n >= 3:
            v1 = detections[-2].velocity if detections[-2].velocity is not None else np.zeros(2)
            v2 = detections[-1].velocity if detections[-1].velocity is not None else np.zeros(2)
            dt = detections[-1].timestamp - detections[-2].timestamp
            if dt > 0:
                acc = (v2 - v1) / dt
                # 정규화 (이미지 크기 기준)
                context.acceleration = acc / np.array([img_w, img_h])
        
        # 2. 궤적 일관성 (직선 피팅 잔차)
        if n >= 3:
            positions = np.array([d.center for d in detections[-5:]])  # 최근 5개
            if len(positions) >= 3:
                # 최소제곱법으로 직선 피팅
                t = np.arange(len(positions))
                try:
                    # x, y 각각 선형 회귀
                    px = np.polyfit(t, positions[:, 0], 1)
                    py = np.polyfit(t, positions[:, 1], 1)
                    
                    # 잔차 계산
                    x_pred = np.polyval(px, t)
                    y_pred = np.polyval(py, t)
                    residuals = np.sqrt((positions[:, 0] - x_pred)**2 + 
                                       (positions[:, 1] - y_pred)**2)
                    
                    # 잔차가 작을수록 일관성 높음 (sigmoid 변환)
                    avg_residual = np.mean(residuals)
                    context.trajectory_consistency = 1.0 / (1.0 + avg_residual / 10.0)
                except np.RankWarning:
                    context.trajectory_consistency = 0.5
        
        # 3. 연속 탐지 프레임 수
        context.consecutive_detections = n
        
        # 4. 평균 confidence
        confidences = [d.confidence for d in detections]
        context.avg_confidence = np.mean(confidences)
        
        # 5. 속도 분산
        velocities = [d.velocity for d in detections if d.velocity is not None]
        if len(velocities) >= 2:
            vel_array = np.array(velocities)
            context.velocity_variance = np.mean(np.var(vel_array, axis=0))
            # 정규화
            context.velocity_variance = min(context.velocity_variance / (img_w * img_h), 1.0)
        
        # 6. 예상 다음 위치 (선형 예측)
        if n >= 2 and detections[-1].velocity is not None:
            context.predicted_next_position = (
                detections[-1].center + detections[-1].velocity * self.dt
            )
        else:
            context.predicted_next_position = detections[-1].center
        
        return context
    
    def get_velocity_history(self, n_frames: int = 3) -> List[np.ndarray]:
        """최근 N 프레임의 속도 히스토리 반환"""
        velocities = []
        for d in list(self.buffer)[-n_frames:]:
            if d.velocity is not None:
                velocities.append(d.velocity)
            else:
                velocities.append(np.zeros(2))
        return velocities
    
    def get_position_history(self, n_frames: int = 5) -> np.ndarray:
        """최근 N 프레임의 위치 히스토리 반환"""
        positions = [d.center for d in list(self.buffer)[-n_frames:]]
        if len(positions) == 0:
            return np.zeros((0, 2))
        return np.array(positions)
    
    def reset(self):
        """버퍼 초기화"""
        self.buffer.clear()
        self.miss_count = 0
    
    def __len__(self) -> int:
        return len(self.buffer)


def extract_temporal_gate_features(
    detection: Dict[str, Any],
    tracker: Any,
    temporal_buffer: TemporalFeatureBuffer,
    image_size: Tuple[int, int]
) -> np.ndarray:
    """Temporal 정보를 포함한 확장 GateNet 피처 추출
    
    기존 10-dim 피처 + Temporal 6-dim = 총 16-dim
    
    Args:
        detection: 현재 프레임 Detection 결과
        tracker: AdaptiveTracker 인스턴스
        temporal_buffer: TemporalFeatureBuffer 인스턴스
        image_size: (width, height)
    
    Returns:
        features: np.ndarray of shape (16,)
    """
    from bolt.gate.network import extract_gate_features
    
    # 기존 10-dim 피처
    base_features = extract_gate_features(detection, tracker, image_size)
    
    # Temporal 6-dim 피처
    temporal_context = temporal_buffer.get_temporal_context(image_size)
    temporal_features = temporal_context.to_feature_vector()
    
    # 결합
    return np.concatenate([base_features, temporal_features])


if __name__ == "__main__":
    print("=== Temporal Feature Buffer Test ===")
    
    buffer = TemporalFeatureBuffer(buffer_size=5, fps=60)
    
    # 시뮬레이션: 직선 궤적
    for i in range(10):
        center = [100 + i * 15, 200 + i * 8]  # 우하향 이동
        confidence = 0.9 - i * 0.02
        buffer.add_detection(frame_id=i, center=center, confidence=confidence)
        
        if i >= 2:
            context = buffer.get_temporal_context()
            print(f"Frame {i}:")
            print(f"  Acceleration: {context.acceleration}")
            print(f"  Trajectory Consistency: {context.trajectory_consistency:.3f}")
            print(f"  Avg Confidence: {context.avg_confidence:.3f}")
            print(f"  Feature Vector: {context.to_feature_vector()}")
            print()
    
    # Miss 테스트
    print("=== Miss Handling ===")
    buffer.add_miss(frame_id=10)
    buffer.add_miss(frame_id=11)
    print(f"Buffer size after misses: {len(buffer)}")
    print(f"Miss count: {buffer.miss_count}")
