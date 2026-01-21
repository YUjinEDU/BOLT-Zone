"""
BOLT-Zone Pitch Type Classifier

MLB Statcast 데이터를 기반으로 학습된 XGBoost 모델을 사용하여
투구된 공의 구종(Pitch Type)을 실시간으로 분류합니다.

Features:
- Velocity (km/h)
- Vertical Break (cm)
- Horizontal Break (cm)
- (Optional) Spin Rate estimate if available

Supported Types:
- FF: 4-Seam Fastball (직구)
- SL: Slider (슬라이더)
- CU: Curveball (커브)
- CH: Changeup (체인지업)
- SI: Sinker (싱커)
- FC: Cutter (커터)
"""

import numpy as np
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger(__name__)

class PitchClassifier:
    def __init__(self, model_path: str = "weights/pitch_classifier.pkl"):
        self.model = None
        self.label_encoder = None
        self.ready = False
        
        self.load_model(model_path)
        
    def load_model(self, model_path: str):
        path = Path(model_path)
        if not path.exists():
            log.warning(f"Pitch Classifier model not found at {path}. System will run without classification.")
            return

        try:
            with open(path, 'rb') as f:
                saved_data = pickle.load(f)
                self.model = saved_data['model']
                self.label_encoder = saved_data['encoder']
                self.features = saved_data.get('features', ['release_speed', 'pfx_x', 'pfx_z'])
            
            self.ready = True
            log.info(f"✅ Pitch Classifier loaded: {len(self.label_encoder.classes_)} types supported")
            
        except Exception as e:
            log.error(f"Failed to load pitch classifier: {e}")

    def classify(self, velocity_kph: float, break_x_cm: float, break_z_cm: float) -> Dict:
        """
        구종 분류 수행
        
        Args:
            velocity_kph: 구속 (km/h)
            break_x_cm: 좌우 무브먼트 (cm) - 포수 시점 (우타자 기준 몸쪽이 -)
            break_z_cm: 상하 무브먼트 (cm) - 중력 보정된 무브먼트 (Vertical Break)
            
        Returns:
            Dict: {'code': 'FF', 'name': 'Four-Seam Fastball', 'prob': 0.95}
        """
        if not self.ready:
            return {'code': 'UNK', 'name': 'Unknown', 'prob': 0.0}

        # MLB 데이터 단위 변환 (Metric -> Imperial might be needed depending on training data)
        # 보통 Statcast 데이터는 MPH, inches 단위가 많음. 확인 후 변환.
        # 여기서는 학습 스크립트에서 Metric으로 변환했다고 가정하거나, Feature Scaling을 맞춤.
        
        # 입력 벡터 (Batch size 1)
        # Feature order must match training: [velocity, break_x, break_z]
        X = np.array([[velocity_kph, break_x_cm, break_z_cm]])
        
        try:
            # XGBoost Inference
            probs = self.model.predict_proba(X)[0]
            best_idx = np.argmax(probs)
            confidence = probs[best_idx]
            
            code = self.label_encoder.inverse_transform([best_idx])[0]
            name = self._get_full_name(code)
            
            return {
                'code': code,
                'name': name,
                'prob': float(confidence)
            }
            
        except Exception as e:
            log.error(f"Classification error: {e}")
            return {'code': 'ERR', 'name': 'Error', 'prob': 0.0}

    def _get_full_name(self, code: str) -> str:
        mapping = {
            'FF': '4-Seam Fastball',
            'SL': 'Slider',
            'CU': 'Curveball',
            'CH': 'Changeup',
            'SI': 'Sinker',
            'FC': 'Cutter',
            'FS': 'Splitter',
            'KC': 'Knuckle Curve'
        }
        return mapping.get(code, code)
