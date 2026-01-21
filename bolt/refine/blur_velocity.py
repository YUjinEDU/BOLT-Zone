"""
OBB to Velocity Estimation

OBB(Oriented Bounding Box)에서 블러 정보를 추출하여 속도를 추정합니다.

모션 블러 원리:
- 블러 길이 = 속도 × 노출 시간
- 블러 각도 = 이동 방향

BlurBall (2025) 논문에서 블러 정보를 활용한 궤적 예측이
MAE 84.4px → 53.0px로 37% 개선됨을 검증.

Reference:
- BlurBall (2025): blur_length = ||velocity|| * exposure_time
"""

import numpy as np
from typing import Tuple, Optional, Union
import logging

log = logging.getLogger(__name__)


def estimate_velocity_from_obb(
    obb: Union[Tuple, np.ndarray],
    exposure_time: float = 1/60.0,
    fps: float = 60.0
) -> np.ndarray:
    """OBB height (blur length)와 angle에서 속도 추정
    
    BlurBall 논문: blur_length = velocity * exposure_time
    
    Args:
        obb: OBB 파라미터 (cx, cy, w, h, angle)
            - cx, cy: 중심 좌표 (pixels)
            - w: 공의 직경 (pixels, 거의 일정)
            - h: 블러 길이 (pixels, 속도에 비례)
            - angle: 블러 각도 (radians, 이동 방향)
        exposure_time: 카메라 노출 시간 (초). 기본값 1/60초
        fps: 비디오 프레임 레이트. 속도를 pixels/frame으로 변환 시 사용
    
    Returns:
        velocity: (vx, vy) - 추정 속도 (pixels/frame)
    
    Example:
        >>> obb = (640, 360, 20, 50, 0.5)  # center, size, angle
        >>> vel = estimate_velocity_from_obb(obb)
        >>> print(f"Velocity: {vel}")
    """
    if len(obb) < 5:
        log.warning(f"Invalid OBB format: expected 5 elements, got {len(obb)}")
        return np.array([0.0, 0.0])
    
    cx, cy, w, h, angle = obb[:5]
    
    # 블러 길이 = OBB height (h)
    # 블러가 없으면 h ≈ w (원형)
    blur_length = max(h - w, 0)  # 순수 블러 길이
    
    if blur_length < 1.0:  # 거의 블러 없음
        return np.array([0.0, 0.0])
    
    # 속도 크기 (pixels/second)
    speed_per_second = blur_length / exposure_time
    
    # 속도 벡터 (pixels/frame)
    speed_per_frame = speed_per_second / fps
    
    vx = speed_per_frame * np.cos(angle)
    vy = speed_per_frame * np.sin(angle)
    
    return np.array([vx, vy])


def estimate_blur_from_velocity(
    velocity: np.ndarray,
    exposure_time: float = 1/60.0,
    fps: float = 60.0
) -> Tuple[float, float]:
    """속도에서 예상 블러 파라미터 추정 (역변환)
    
    GateNet 입력 피처 생성에 사용.
    
    Args:
        velocity: (vx, vy) 속도 (pixels/frame)
        exposure_time: 카메라 노출 시간 (초)
        fps: 비디오 프레임 레이트
    
    Returns:
        (blur_length, blur_angle): 예상 블러 길이(pixels)와 각도(radians)
    """
    vx, vy = velocity[:2] if len(velocity) >= 2 else (velocity[0], 0)
    
    # 속도 크기 (pixels/frame → pixels/second)
    speed_per_frame = np.sqrt(vx**2 + vy**2)
    speed_per_second = speed_per_frame * fps
    
    # 블러 길이
    blur_length = speed_per_second * exposure_time
    
    # 블러 각도
    if speed_per_frame > 0.01:
        blur_angle = np.arctan2(vy, vx)
    else:
        blur_angle = 0.0
    
    return blur_length, blur_angle


def refine_trajectory_with_blur(
    positions: np.ndarray,
    blur_lengths: np.ndarray,
    blur_angles: np.ndarray,
    exposure_time: float = 1/60.0,
    position_weight: float = 1.0,
    blur_weight: float = 0.2
) -> np.ndarray:
    """블러 정보를 활용한 궤적 정제
    
    BlurBall 논문의 trajectory prediction 방법론.
    위치와 블러 정보를 결합하여 더 정확한 궤적 추정.
    
    Args:
        positions: (N, 2) 배열, 각 프레임의 탐지된 (x, y) 위치
        blur_lengths: (N,) 배열, 각 프레임의 블러 길이
        blur_angles: (N,) 배열, 각 프레임의 블러 각도
        exposure_time: 카메라 노출 시간 (초)
        position_weight: 위치 손실 가중치
        blur_weight: 블러 손실 가중치
    
    Returns:
        refined_positions: (N, 2) 정제된 위치 배열
    
    Reference:
        BlurBall 논문: position + blur 사용 시 MAE 37% 개선
    """
    n = len(positions)
    if n < 3:
        return positions.copy()
    
    # 간단한 방법: 블러 방향으로 속도 추정하여 보정
    refined = positions.copy()
    
    for i in range(1, n - 1):
        if blur_lengths[i] > 1.0:  # 블러가 있는 경우
            # 블러에서 추정한 속도
            speed = blur_lengths[i] / exposure_time
            vx = speed * np.cos(blur_angles[i])
            vy = speed * np.sin(blur_angles[i])
            
            # 이전/다음 프레임과의 일관성 확인
            actual_vx = (positions[i+1, 0] - positions[i-1, 0]) / 2
            actual_vy = (positions[i+1, 1] - positions[i-1, 1]) / 2
            
            # 블러 방향과 실제 이동 방향의 내적
            dot_prod = vx * actual_vx + vy * actual_vy
            
            # 방향이 반대(180도 차이)인 경우 벡터 뒤집기
            if dot_prod < 0:
                vx = -vx
                vy = -vy
                # 뒤집은 후 다시 내적 계산 (검증용)
                dot_prod = -dot_prod
            
            # 방향이 일관적인 경우에만 보정 수행 (임계값: 0보다 큼)
            if dot_prod > 0:
                # 블러 기반 보정
                blend = blur_weight / (position_weight + blur_weight)
                predicted_x = positions[i-1, 0] + vx * exposure_time
                predicted_y = positions[i-1, 1] + vy * exposure_time
                refined[i, 0] = (1 - blend) * positions[i, 0] + blend * predicted_x
                refined[i, 1] = (1 - blend) * positions[i, 1] + blend * predicted_y
    
    return refined


def obb_to_blur_params(obb: Union[Tuple, np.ndarray]) -> Tuple[float, float, float, float]:
    """OBB에서 블러 파라미터 추출
    
    Args:
        obb: (cx, cy, w, h, angle) OBB 파라미터
    
    Returns:
        (center_x, center_y, blur_length, blur_angle)
    """
    cx, cy, w, h, angle = obb[:5]
    blur_length = max(h - w, 0)  # 순수 블러 길이
    return cx, cy, blur_length, angle


if __name__ == "__main__":
    print("=== Blur Velocity Estimation Test ===")
    
    # OBB → Velocity 테스트
    obb = (640, 360, 20, 70, 0.5)  # 블러 길이 50px, 각도 0.5rad
    velocity = estimate_velocity_from_obb(obb, exposure_time=1/60.0)
    print(f"OBB: {obb}")
    print(f"Estimated velocity: {velocity} pixels/frame")
    print(f"Speed: {np.linalg.norm(velocity):.2f} pixels/frame")
    
    # Velocity → Blur 역변환 테스트
    blur_len, blur_angle = estimate_blur_from_velocity(velocity)
    print(f"\nReverse estimation:")
    print(f"Blur length: {blur_len:.2f} pixels")
    print(f"Blur angle: {blur_angle:.2f} radians")
    
    # 궤적 정제 테스트
    print("\n=== Trajectory Refinement Test ===")
    positions = np.array([
        [100, 200],
        [150, 210],
        [200, 220],
        [250, 235],
        [300, 250]
    ], dtype=float)
    blur_lengths = np.array([0, 50, 45, 40, 0])
    blur_angles = np.array([0, 0.2, 0.25, 0.3, 0])
    
    refined = refine_trajectory_with_blur(positions, blur_lengths, blur_angles)
    print(f"Original positions:\n{positions}")
    print(f"Refined positions:\n{refined}")
    print(f"Difference:\n{refined - positions}")
