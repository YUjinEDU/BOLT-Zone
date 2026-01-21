"""
BOLT-Gate Network (GateNet) v2

Rule-based Gating을 대체하는 초경량 MLP 모델.
Detect 결과와 트래킹 상태 정보를 입력받아 Refine 수행 여부를 판단합니다.

개선사항 (v2):
- ✅ 입력 피처 6 → 10 확장
- ✅ 모델 구조 조정 (10 → 32 → 16 → 1)
- ✅ extract_gate_features 유틸 함수 추가

Features (Input Dim: 10):
1. Conf (1): Detect confidence
2. Width (1): Normalized bbox width
3. Height (1): Normalized bbox height  
4. Velocity X (1): Normalized velocity x
5. Velocity Y (1): Normalized velocity y
6. Uncertainty (1): Tracker covariance trace
7. Miss Count (1): 연속 탐지 실패 횟수 (NEW)
8. Expected Blur Length (1): 속도 기반 예상 블러 길이 (NEW)
9. Edge Distance (1): 화면 가장자리까지 거리 (NEW)
10. Mode Probability CA (1): CA(변화구) 모드 확률 (NEW)

Architecture:
Lightweight MLP: Input(10) -> FC(32) -> ReLU -> FC(16) -> ReLU -> FC(1) -> Sigmoid
FLOPS가 거의 없어 CPU 오버헤드 무시 가능 수준.

Reference:
- DGNets (ICCV 2021): Dynamic Dual Gating Neural Networks
- Adaptive Inference (arXiv 1702.07811)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging

log = logging.getLogger(__name__)


class GateNet(nn.Module):
    """Adaptive Gating Network for BOLT-Refine
    
    Detect 결과의 불확실성과 트래킹 상태를 종합하여
    Refine(OBB 정밀 분석) 수행 여부를 결정합니다.
    
    입력 피처 (10-dim):
    - conf: Detection confidence
    - w, h: Normalized bbox width/height
    - vx, vy: Normalized velocity
    - uncertainty: Tracker covariance trace
    - miss_count: Consecutive miss count
    - expected_blur: Expected blur length
    - edge_dist: Distance to frame edge
    - mode_prob: CA model probability (Ablation에서만 사용)
    """
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        
        # 개선된 모델 구조 (10 → 32 → 16 → 1)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def predict(self, features: torch.Tensor, threshold: float = 0.5) -> bool:
        """
        Inference helper
        Returns True if Refine is needed
        """
        with torch.no_grad():
            prob = self.forward(features)
            return prob.item() > threshold
    
    def get_refine_probability(self, features: torch.Tensor) -> float:
        """Refine이 필요할 확률 반환 (0.0 ~ 1.0)"""
        with torch.no_grad():
            return self.forward(features).item()


def extract_gate_features(
    detection: Dict[str, Any],
    tracker: Any,
    image_size: Tuple[int, int],
    exposure_time: float = 1/60.0
) -> np.ndarray:
    """Detection 결과와 Tracker 상태에서 GateNet 입력 피처 추출
    
    Args:
        detection: YOLO detection 결과 딕셔너리
            - 'confidence': float (0.0 ~ 1.0)
            - 'bbox': [x1, y1, x2, y2] or [cx, cy, w, h]
            - 'center': [cx, cy] (optional)
        tracker: AdaptiveTracker 인스턴스
        image_size: (width, height) 이미지 크기
        exposure_time: 카메라 노출 시간 (초)
    
    Returns:
        features: np.ndarray of shape (10,)
            [conf, width, height, vx, vy, uncertainty, 
             miss_count, expected_blur, edge_dist, mode_prob_ca]
    """
    img_w, img_h = image_size
    
    # 1. Confidence
    conf = detection.get('confidence', 0.0)
    
    # 2-3. Normalized Width, Height
    bbox = detection.get('bbox', [0, 0, 0, 0])
    if len(bbox) == 4:
        if bbox[2] > bbox[0]:  # x1, y1, x2, y2 format
            width = (bbox[2] - bbox[0]) / img_w
            height = (bbox[3] - bbox[1]) / img_h
        else:  # cx, cy, w, h format
            width = bbox[2] / img_w
            height = bbox[3] / img_h
    else:
        width, height = 0.0, 0.0
    
    # 4-5. Normalized Velocity
    velocity = tracker.get_velocity() if hasattr(tracker, 'get_velocity') else np.zeros(3)
    vx = velocity[0] / img_w  # Normalized
    vy = velocity[1] / img_h
    
    # 6. Uncertainty (Tracker covariance trace)
    if hasattr(tracker, 'filters'):
        # IMM: 첫 번째 필터의 covariance
        uncertainty = np.trace(tracker.filters[0].P[:3, :3]) / 100.0
    elif hasattr(tracker, 'P'):
        uncertainty = np.trace(tracker.P[:3, :3]) / 100.0
    else:
        uncertainty = 1.0
    uncertainty = min(uncertainty, 10.0)  # Clamp
    
    # 7. Miss Count (연속 탐지 실패 횟수)
    miss_count = getattr(tracker, 'miss_count', 0) / 3.0  # Normalized by MAX_MISS
    
    # 8. Expected Blur Length
    if hasattr(tracker, 'get_expected_blur_length'):
        expected_blur = tracker.get_expected_blur_length(exposure_time) / img_w
    else:
        speed = np.linalg.norm(velocity[:2])
        expected_blur = speed * exposure_time / img_w
    
    # 9. Edge Distance (화면 가장자리까지 최소 거리)
    center = detection.get('center', None)
    if center is None and len(bbox) == 4:
        if bbox[2] > bbox[0]:
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        else:
            center = [bbox[0], bbox[1]]
    
    if center is not None:
        cx, cy = center
        edge_dist_x = min(cx, img_w - cx) / img_w
        edge_dist_y = min(cy, img_h - cy) / img_h
        edge_dist = min(edge_dist_x, edge_dist_y)
    else:
        edge_dist = 0.5  # Center
    
    # 10. Mode Probability (CA/변화구 확률)
    if hasattr(tracker, 'get_mode_probability'):
        _, mode_prob_ca = tracker.get_mode_probability()
    else:
        mode_prob_ca = 0.5
    
    features = np.array([
        conf,           # 0: Confidence
        width,          # 1: Width
        height,         # 2: Height
        vx,             # 3: Velocity X
        vy,             # 4: Velocity Y
        uncertainty,    # 5: Uncertainty
        miss_count,     # 6: Miss Count
        expected_blur,  # 7: Expected Blur Length
        edge_dist,      # 8: Edge Distance
        mode_prob_ca    # 9: Mode Probability (CA)
    ], dtype=np.float32)
    
    return features


def export_gate_model(model_path: str, input_dim: int = 10):
    """
    ONNX Export Helper for GateNet
    """
    model = GateNet(input_dim=input_dim)
    model.eval()
    
    dummy_input = torch.randn(1, input_dim)
    output_path = model_path.replace('.pt', '.onnx')
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=11
        )
        log.info(f"GateNet exported to {output_path}")
    except Exception as e:
        log.error(f"Failed to export GateNet: {e}")


if __name__ == "__main__":
    print("=== GateNet v2 Test ===")
    
    # 모델 테스트
    net = GateNet(input_dim=10)
    print(f"Model Parameters: {sum(p.numel() for p in net.parameters())}")
    
    dummy = torch.randn(5, 10)
    out = net(dummy)
    print(f"Output shape: {out.shape}")
    print(f"Output: {out.squeeze()}")
    
    # Feature Extraction 테스트
    print("\n=== Feature Extraction Test ===")
    from bolt.track.adaptive_tracker import AdaptiveTracker
    
    tracker = AdaptiveTracker()
    # 시뮬레이션 업데이트
    tracker.update(np.array([640, 360, 0]), confidence=0.9)
    tracker.update(np.array([650, 365, 0]), confidence=0.85)
    
    detection = {
        'confidence': 0.85,
        'bbox': [630, 350, 670, 380],  # x1, y1, x2, y2
        'center': [650, 365]
    }
    
    features = extract_gate_features(detection, tracker, (1280, 720))
    print(f"Feature dim: {len(features)}")
    print(f"Features: {features}")
    
    # GateNet 예측
    features_tensor = torch.from_numpy(features).unsqueeze(0)
    prob = net.get_refine_probability(features_tensor)
    print(f"Refine probability: {prob:.4f}")

