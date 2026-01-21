"""
BOLT-Zone Calibration Module

카메라의 2D 영상 좌표를 현실 세계의 3D 좌표로 변환(PnP)하는 핵심 모듈.
Hybrid 방식을 사용하여 다양한 환경에서 강인한 Calibration을 지원합니다.

Modes:
1. **AI Auto (Marker-less)**: YOLO-Seg로 홈 플레이트를 찾아 5개 꼭짓점 추출 (Best)
2. **ArUco (Marker)**: ArUco 마커(Dict 4X4) 4개 코너 추출 (Fallback)
3. **Manual**: 사용자가 직접 4개 점 클릭 (Last Resort)

Home Plate Specification (Official):
- Width: 17 inches (43.18 cm)
- Side Length: 8.5 inches (21.59 cm) -> 앞쪽 빗변
- Height: 17 inches (to the tip)
"""

import cv2
import numpy as np
import logging
from typing import Tuple, List, Optional, Dict

log = logging.getLogger(__name__)


class HybridCalibrator:
    # 홈 플레이트 3D 좌표 (단위: cm)
    # 원점(0,0,0)은 홈 플레이트 뒷쪽 뾰족한 끝 (Tip)
    # 좌표계: x(우), y(전방), z(상)
    HOME_PLATE_3D = np.array([
        [0, 0, 0],              # 0: Tip (Bottom)
        [-21.59, 21.59, 0],     # 1: Left Corner
        [-21.59, 43.18, 0],     # 2: Top-Left
        [21.59, 43.18, 0],      # 3: Top-Right
        [21.59, 21.59, 0]       # 4: Right Corner
    ], dtype=np.float32)

    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rvec = None
        self.tvec = None
        self.mode = None  # 'AI', 'ArUco', 'Manual'
        
        # YOLO-Seg 모델 (나중에 로드)
        self.seg_model = None

    def load_ai_model(self, model_path: str):
        """YOLO-Seg 모델 로드 (Ultralytics)"""
        from ultralytics import YOLO
        try:
            self.seg_model = YOLO(model_path)
            log.info(f"Calibration AI Model loaded: {model_path}")
        except Exception as e:
            log.warning(f"Failed to load AI model: {e}")

    def calibrate(self, frame: np.ndarray) -> bool:
        """
        Hybrid Calibration Pipeline
        Priority: AI -> ArUco -> False
        """
        # 1. AI Auto Mode (Home Plate)
        if self.seg_model:
            success = self._calibrate_with_ai(frame)
            if success:
                self.mode = 'AI'
                log.info("✅ Calibration Success (AI - Home Plate)")
                return True

        # 2. ArUco Mode
        success = self._calibrate_with_aruco(frame)
        if success:
            self.mode = 'ArUco'
            log.info("✅ Calibration Success (ArUco Marker)")
            return True

        log.warning("❌ Calibration Failed. Try Manual Mode.")
        return False

    def _calibrate_with_ai(self, frame: np.ndarray) -> bool:
        """YOLO-Seg 기반 홈 플레이트 검출"""
        if self.seg_model is None:
            return False
            
        results = self.seg_model(frame, verbose=False)
        if not results or not results[0].masks:
            return False
            
        # 가장 신뢰도 높은 마스크 선택 (Class 0: Home Plate 가정)
        # 실제 구현 시: 마스크에서 Contour 추출 -> approxPolyDP -> 5개 꼭짓점 찾기
        mask = results[0].masks.data[0].cpu().numpy()
        
        # Placeholder: 마스크 처리 로직 (복잡하므로 간소화)
        # contours, _ = cv2.findContours(...)
        # approx = cv2.approxPolyDP(...)
        # if len(approx) == 5: ...
        
        # (TODO: Segmentation Post-processing 구현 필요)
        return False 

    def _calibrate_with_aruco(self, frame: np.ndarray) -> bool:
        """ArUco 마커 기반"""
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
        
        if ids is not None and len(corners) > 0:
            # 가정: 4개의 ArUco 마커가 홈 플레이트 주변에 배치됨
            # 여기서는 단순화를 위해 1개의 마커(ID 0)가 홈 플레이트 중심에 있다고 가정
            # 실제로는 4개 마커의 3D 좌표 정의 필요
            
            # Example: ID 0 is at (0, 20, 0)
            obj_points = np.array([
                [-5, 20, 0], [5, 20, 0], [5, 20+10, 0], [-5, 20+10, 0]
            ], dtype=np.float32)
            
            img_points = corners[0][0] # First marker
            
            ret, self.rvec, self.tvec = cv2.solvePnP(
                obj_points, img_points, self.camera_matrix, self.dist_coeffs
            )
            return ret
            
        return False

    def project_points(self, points_3d: np.ndarray) -> np.ndarray:
        """3D -> 2D 투영 (시각화용)"""
        if self.rvec is None or self.tvec is None:
            return None
            
        img_points, _ = cv2.projectPoints(
            points_3d, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs
        )
        return img_points.reshape(-1, 2)

    def back_project(self, point_2d: np.ndarray, z: float = 0) -> np.ndarray:
        """2D -> 3D 역투영 (지면 z=0 가정)"""
        # (TODO: ray-plane intersection 구현)
        pass
