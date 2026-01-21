"""
BOLT-Zone Physics Engine

야구공의 비행 궤적을 물리 법칙에 기반하여 모델링합니다.
중력, 공기저항, 마그누스 효과를 포함합니다.

⚠️ 중요 (LIMITATIONS):
    이 물리 모델은 **"참고용(Reference)"**이지 **"예측용(Prediction)"**이 아닙니다.
    
    1. **스핀 정보 부재**: 단일 2D 카메라로는 공의 회전(Spin)을 측정할 수 없습니다.
       Spin을 알 수 없으므로, 마그누스 효과 계산은 추정치에 불과합니다.
       
    2. **Seam-Shifted Wake (SSW)**: 야구공 실밥(Seam)에 의한 추가 변화(최대 22cm)는
       이 물리 모델로는 계산할 수 없습니다.
       
    3. **결론**: 이 모델만으로 프로 선수의 투구 궤적을 정확히 예측하는 것은 불가능합니다.
       MLB가 Hawk-Eye(±0.1인치 정확도)를 사용하는 이유입니다.

    **본 모듈의 용도**:
    - 합성 데이터 생성 시 "그럴듯한" 궤적 생성 (Data Augmentation)
    - 물리적 특성(낙차, 변화량) 추정을 위한 참고값 제공
"""

import numpy as np
from typing import Tuple
from scipy.optimize import least_squares
import logging

log = logging.getLogger(__name__)

class BaseballPhysics:
    # Constants
    G = 9.81          # Gravity (m/s^2)
    RHO = 1.225       # Air density (kg/m^3)
    MASS = 0.145      # Mass (kg)
    DIAMETER = 0.074  # Diameter (m)
    AREA = np.pi * (DIAMETER / 2) ** 2
    
    CD_DEFAULT = 0.30  # Drag Coefficient
    CL_DEFAULT = 0.15  # Lift Coefficient (Highly variable in reality!)

    def __init__(self):
        self.const_drag = 0.5 * self.RHO * self.AREA / self.MASS
        self.const_lift = 0.5 * self.RHO * self.AREA / self.MASS

    def get_acceleration(
        self, 
        velocity: np.ndarray, 
        cd: float = None, 
        spin_vec: np.ndarray = None, 
        cl: float = None
    ) -> np.ndarray:
        if cd is None: cd = self.CD_DEFAULT
        if cl is None: cl = self.CL_DEFAULT
        
        v_mag = np.linalg.norm(velocity)
        if v_mag == 0: return np.array([0, 0, -self.G])
        
        acc_g = np.array([0, 0, -self.G])
        acc_d = - (self.const_drag * cd * v_mag) * velocity
        
        acc_m = np.array([0., 0., 0.])
        if spin_vec is not None and np.linalg.norm(spin_vec) > 0:
            w_x_v = np.cross(spin_vec, velocity)
            norm_wxv = np.linalg.norm(w_x_v)
            if norm_wxv > 0:
                dir_m = w_x_v / norm_wxv
                mag_m = self.const_lift * cl * (v_mag**2)
                acc_m = mag_m * dir_m

        return acc_g + acc_d + acc_m

    def simulate_trajectory(
        self,
        start_pos: np.ndarray,
        start_vel: np.ndarray,
        times: np.ndarray,
        cd: float = None,
        spin_vec: np.ndarray = None,
        cl: float = None
    ) -> np.ndarray:
        """
        Runge-Kutta 4th Order Simulation.
        
        ⚠️ 이 결과는 "참고용(Reference)"입니다.
        실제 야구공 궤적과 최대 22cm까지 차이가 발생할 수 있습니다.
        """
        dt = 0.001
        max_t = np.max(times)
        
        pos = start_pos.copy()
        vel = start_vel.copy()
        
        results = np.zeros((len(times), 3))
        
        t_current = 0.0
        target_indices = np.argsort(times)
        curr_idx = 0
        
        while t_current <= max_t + dt:
            while curr_idx < len(times) and t_current >= times[target_indices[curr_idx]]:
                results[target_indices[curr_idx]] = pos
                curr_idx += 1
            if curr_idx >= len(times): break
            
            def acc_fn(v):
                return self.get_acceleration(v, cd, spin_vec, cl)
                
            k1_v = acc_fn(vel) * dt
            k1_p = vel * dt
            k2_v = acc_fn(vel + 0.5*k1_v) * dt
            k2_p = (vel + 0.5*k1_v) * dt
            k3_v = acc_fn(vel + 0.5*k2_v) * dt
            k3_p = (vel + 0.5*k2_v) * dt
            k4_v = acc_fn(vel + k3_v) * dt
            k4_p = (vel + k3_v) * dt
            
            vel += (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
            pos += (k1_p + 2*k2_p + 2*k3_p + k4_p) / 6
            t_current += dt
            
        return results
