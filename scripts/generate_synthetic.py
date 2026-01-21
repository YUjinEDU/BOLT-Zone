"""
Physics-Guided Synthetic Data Generator (Enhanced v2)

물리 엔진(BaseballPhysics)을 활용하여 사실적인 Motion Blur가 포함된
야구공 투구 이미지를 합성하고, 자동으로 OBB 라벨을 생성하는 스크립트입니다.

Enhancements (v2):
- ✅ 실제 야구장 배경 이미지 통합 (다양한 환경: 낮/밤/비/실내)
- ✅ Gaussian Blur 적용으로 자연스러운 모션 블러
- ✅ 공 텍스처 패치 추가로 사실성 향상
- ✅ 투구 방향 제어 (Strike Zone 통과 여부)
- ✅ 자유로운 카메라 위치/방향 설정

Process:
1. Physics: 다양한 초기 조건으로 3D 궤적 시뮬레이션 (투구 방향 제어)
2. Camera: 자유로운 카메라 위치/방향으로 Look-At 매트릭스 생성
3. Projection: 3D 궤적 -> 2D 이미지 평면 투영
4. Rendering: 공 텍스처 + Gaussian Blur로 사실적인 모션 블러 렌더링
5. Labeling: OBB 자동 계산 및 저장

Usage:
    # 기본 사용 (Umpire View)
    python scripts/generate_synthetic.py --num 100 --output data/synthetic
    
    # 배경 이미지 폴더 지정
    python scripts/generate_synthetic.py --num 100 --backgrounds data/backgrounds --output data/synthetic
    
    # 카메라 위치 커스터마이징 (x, y, z)
    python scripts/generate_synthetic.py --num 50 --camera-pos 2 -3 2 --output data/synthetic_side
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import random

from bolt.track.physics import BaseballPhysics


def create_lookat_matrix(camera_pos, target_pos, up_vector=np.array([0, 0, 1])):
    """
    Look-At 매트릭스 생성 (Extrinsic Matrix)
    
    Args:
        camera_pos: 카메라 위치 [x, y, z]
        target_pos: 카메라가 바라보는 지점 [x, y, z]
        up_vector: 카메라의 위쪽 방향 벡터 (기본: Z축)
    
    Returns:
        R: 3x3 회전 매트릭스
        t: 3x1 이동 벡터
    """
    # Z-axis (Forward): Camera -> Target
    z_axis = target_pos - camera_pos
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # X-axis (Right): Up x Z
    x_axis = np.cross(up_vector, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Y-axis (Down): Z x X
    y_axis = np.cross(z_axis, x_axis)
    
    # Rotation matrix (World -> Camera)
    R = np.vstack([x_axis, y_axis, z_axis])
    
    # Translation vector
    t = -R @ camera_pos
    
    return R, t


class SyntheticGenerator:
    def __init__(self, width=1280, height=720, background_dir=None, 
                 camera_pos=None, look_at=None, focal_length=1000):
        """
        Args:
            width, height: 이미지 크기
            background_dir: 배경 이미지 폴더 경로 (None이면 검은 배경)
            camera_pos: 카메라 위치 [x, y, z] (기본: Umpire View)
            look_at: 카메라가 바라보는 지점 [x, y, z] (기본: 홈플레이트)
            focal_length: 카메라 초점 거리 (픽셀 단위)
        """
        self.width = width
        self.height = height
        self.physics = BaseballPhysics()
        
        # 배경 이미지 로드
        self.backgrounds = []
        if background_dir and Path(background_dir).exists():
            bg_path = Path(background_dir)
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                self.backgrounds.extend(list(bg_path.glob(ext)))
            print(f"✅ Loaded {len(self.backgrounds)} background images")
        else:
            print("⚠️  No background images found. Using black background.")
        
        # 카메라 설정
        if camera_pos is None:
            # 기본: Umpire View (홈플레이트 뒤 2m, 높이 1.5m)
            camera_pos = np.array([0, -2.0, 1.5])
        else:
            camera_pos = np.array(camera_pos)
        
        if look_at is None:
            # 기본: 홈플레이트 중앙 (높이 1m)
            look_at = np.array([0, 0, 1.0])
        else:
            look_at = np.array(look_at)
        
        # Intrinsic Matrix
        self.K = np.array([
            [focal_length, 0, width/2],
            [0, focal_length, height/2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Extrinsic Matrix (Look-At)
        self.R, self.t = create_lookat_matrix(camera_pos, look_at)
        
        print(f"📷 Camera Position: {camera_pos}")
        print(f"🎯 Looking At: {look_at}")
        
    def generate_trajectory(self, target_zone='random'):
        """
        물리 기반 3D 궤적 생성 (투구 방향 제어)
        
        Args:
            target_zone: 'strike', 'ball', 'random' 중 하나
        
        Returns:
            traj_3d: Nx3 배열 (3D 궤적)
        """
        # Strike Zone 범위 (홈플레이트 기준)
        # X: -0.22 ~ 0.22m (17인치 = 0.43m)
        # Z: 0.5 ~ 1.1m (무릎~가슴)
        
        # Pitcher Mound 위치
        pitcher_x = np.random.normal(0, 0.2)
        pitcher_y = 18.44  # 투수판 거리
        pitcher_z = np.random.normal(1.8, 0.1)
        
        # Target 위치 (홈플레이트)
        if target_zone == 'strike':
            # Strike Zone 내부
            target_x = np.random.uniform(-0.22, 0.22)
            target_z = np.random.uniform(0.5, 1.1)
        elif target_zone == 'ball':
            # Strike Zone 외부
            if random.random() < 0.5:
                # 좌우로 벗어남
                target_x = np.random.uniform(-0.5, 0.5)
                target_z = np.random.uniform(0.3, 1.3)
            else:
                # 상하로 벗어남
                target_x = np.random.uniform(-0.22, 0.22)
                target_z = np.random.choice([
                    np.random.uniform(0.2, 0.5),   # 낮음
                    np.random.uniform(1.1, 1.5)    # 높음
                ])
        else:  # random
            target_x = np.random.uniform(-0.5, 0.5)
            target_z = np.random.uniform(0.3, 1.3)
        
        target_y = 0  # 홈플레이트
        
        # 초기 속도 계산 (투수 -> 타겟)
        speed = np.random.uniform(36, 44)  # 130~160 km/h
        flight_time = pitcher_y / speed  # 대략적인 비행 시간
        
        vx = (target_x - pitcher_x) / flight_time
        vy = -speed  # 홈플레이트 방향
        vz = (target_z - pitcher_z) / flight_time + 0.5 * 9.81 * flight_time  # 중력 보정
        
        pos_start = np.array([pitcher_x, pitcher_y, pitcher_z])
        vel_start = np.array([vx, vy, vz])
        
        # 물리 시뮬레이션 (1/60초 동안 10 프레임)
        times = np.linspace(0, 1/60.0, 10)
        traj_3d = self.physics.simulate_trajectory(pos_start, vel_start, times)
        
        return traj_3d
    
    def project_to_image(self, points_3d):
        """
        3D (World) -> 2D (Image) Projection
        
        Args:
            points_3d: Nx3 배열 (World 좌표계)
        
        Returns:
            points_2d: Nx2 배열 (Image 좌표계)
        """
        points_2d = []
        
        for p in points_3d:
            # World -> Camera 좌표계 변환
            p_cam = self.R @ p + self.t
            
            # Camera 좌표계: Z가 깊이 (Forward)
            x_c, y_c, z_c = p_cam
            
            if z_c <= 0.1:  # 카메라 뒤쪽은 무시
                continue
            
            # Perspective Projection
            u = self.K[0, 0] * (x_c / z_c) + self.K[0, 2]
            v = self.K[1, 1] * (y_c / z_c) + self.K[1, 2]
            
            points_2d.append([u, v])
        
        return np.array(points_2d) if points_2d else np.array([])
    
    def create_ball_texture(self, diameter=20):
        """
        야구공 텍스처 생성 (흰색 원 + 실밥 패턴)
        
        Args:
            diameter: 공 지름 (픽셀)
        
        Returns:
            ball_img: 공 이미지 (RGBA)
            ball_mask: 알파 마스크
        """
        size = diameter
        ball_img = np.zeros((size, size, 4), dtype=np.uint8)
        
        # 흰색 원
        center = (size // 2, size // 2)
        radius = size // 2
        cv2.circle(ball_img, center, radius, (240, 240, 240, 255), -1)
        
        # 간단한 실밥 패턴 (빨간 곡선)
        cv2.ellipse(ball_img, center, (radius-2, radius//2), 45, 0, 180, (200, 60, 60, 255), 1)
        cv2.ellipse(ball_img, center, (radius-2, radius//2), -45, 0, 180, (200, 60, 60, 255), 1)
        
        # 알파 마스크
        ball_mask = ball_img[:, :, 3]
        
        return ball_img, ball_mask
    
    def draw_motion_blur(self, img, points_2d, speed_factor=1.0):
        """
        Gaussian Blur를 활용한 사실적인 모션 블러 렌더링
        
        Args:
            img: 배경 이미지
            points_2d: 2D 궤적 포인트
            speed_factor: 속도 (블러 강도 제어)
        
        Returns:
            img: 렌더링된 이미지
            obb_label: OBB 라벨 [class, cx, cy, w, h, angle_rad]
        """
        if len(points_2d) < 2:
            return img, None
        
        # 궤적 시작/끝점
        pt1 = points_2d[0].astype(int)
        pt2 = points_2d[-1].astype(int)
        
        # 화면 밖이면 스킵
        if (pt1[0] < 0 or pt1[0] >= self.width or pt1[1] < 0 or pt1[1] >= self.height or
            pt2[0] < 0 or pt2[0] >= self.width or pt2[1] < 0 or pt2[1] >= self.height):
            return img, None
        
        # OBB 계산
        center = (pt1 + pt2) / 2
        length = np.linalg.norm(pt2 - pt1)
        
        # 공 크기 (거리에 따라 변함)
        ball_diameter = max(10, min(30, int(20 * speed_factor)))
        width = ball_diameter
        
        angle_deg = np.degrees(np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))
        angle_rad = np.radians(angle_deg)
        
        # 1. 공 텍스처 생성
        ball_img, ball_mask = self.create_ball_texture(ball_diameter)
        
        # 2. 궤적을 따라 공 배치 (여러 개 겹쳐서 블러 효과)
        overlay = np.zeros_like(img, dtype=np.float32)
        
        # 궤적 보간 (더 많은 포인트 생성)
        num_blur_points = max(3, int(length / 5))
        for i in range(num_blur_points):
            t = i / max(1, num_blur_points - 1)
            pos = pt1 * (1 - t) + pt2 * t
            x, y = int(pos[0]), int(pos[1])
            
            # 공 패치 붙이기
            half_d = ball_diameter // 2
            x1, y1 = max(0, x - half_d), max(0, y - half_d)
            x2, y2 = min(self.width, x + half_d), min(self.height, y + half_d)
            
            bx1, by1 = half_d - (x - x1), half_d - (y - y1)
            bx2, by2 = half_d + (x2 - x), half_d + (y2 - y)
            
            if x2 > x1 and y2 > y1 and bx2 > bx1 and by2 > by1:
                ball_patch = ball_img[by1:by2, bx1:bx2, :3]
                mask_patch = ball_mask[by1:by2, bx1:bx2] / 255.0
                
                for c in range(3):
                    overlay[y1:y2, x1:x2, c] += ball_patch[:, :, c] * mask_patch * 0.3
        
        # 3. Gaussian Blur 적용
        blur_ksize = max(3, int(length / 10) * 2 + 1)  # 홀수
        overlay = cv2.GaussianBlur(overlay, (blur_ksize, blur_ksize), 0)
        
        # 4. 배경과 합성
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        img = cv2.addWeighted(img, 1.0, overlay, 0.7, 0)
        
        # OBB 라벨 (YOLO OBB 포맷)
        obb_label = [0, center[0], center[1], width, length, angle_rad]
        
        return img, obb_label
    
    def run(self, num_samples, out_dir):
        """
        합성 데이터 생성 실행
        
        Args:
            num_samples: 생성할 샘플 수
            out_dir: 출력 디렉토리
        """
        out_path = Path(out_dir)
        (out_path / "images").mkdir(parents=True, exist_ok=True)
        (out_path / "labels").mkdir(parents=True, exist_ok=True)
        
        successful = 0
        
        for i in tqdm(range(num_samples), desc="Generating Synthetic Data"):
            # 1. 배경 이미지 로드
            if self.backgrounds:
                bg_path = random.choice(self.backgrounds)
                bg_img = cv2.imread(str(bg_path))
                if bg_img is not None:
                    img = cv2.resize(bg_img, (self.width, self.height))
                else:
                    img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            else:
                # 검은 배경
                img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # 2. 궤적 생성 (Strike/Ball 랜덤)
            zone = random.choice(['strike', 'ball', 'random'])
            traj_3d = self.generate_trajectory(target_zone=zone)
            
            # 3. 2D 투영
            traj_2d = self.project_to_image(traj_3d)
            
            if len(traj_2d) < 2:
                continue
            
            # 4. 모션 블러 렌더링
            speed = np.linalg.norm(traj_3d[0] - traj_3d[-1])
            img, label = self.draw_motion_blur(img, traj_2d, speed_factor=speed/5.0)
            
            if label is not None:
                # 저장
                cv2.imwrite(str(out_path / "images" / f"syn_{i:04d}.jpg"), img)
                
                # YOLO OBB 라벨 (정규화)
                ln = [
                    0,  # class
                    label[1] / self.width,   # center_x
                    label[2] / self.height,  # center_y
                    label[3] / self.width,   # width
                    label[4] / self.height,  # height
                    label[5]  # angle (radians)
                ]
                
                with open(out_path / "labels" / f"syn_{i:04d}.txt", "w") as f:
                    f.write(f"{int(ln[0])} {ln[1]:.6f} {ln[2]:.6f} {ln[3]:.6f} {ln[4]:.6f} {ln[5]:.6f}\n")
                
                successful += 1
        
        print(f"\n✅ Successfully generated {successful}/{num_samples} samples")
        print(f"📁 Saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physics-Based Synthetic Data Generator")
    parser.add_argument("--num", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--backgrounds", type=str, default=None, help="Background images directory")
    parser.add_argument("--width", type=int, default=1280, help="Image width")
    parser.add_argument("--height", type=int, default=720, help="Image height")
    parser.add_argument("--camera-pos", type=float, nargs=3, default=None, 
                        metavar=('X', 'Y', 'Z'), help="Camera position (x, y, z)")
    parser.add_argument("--look-at", type=float, nargs=3, default=None,
                        metavar=('X', 'Y', 'Z'), help="Look-at target (x, y, z)")
    parser.add_argument("--focal-length", type=int, default=1000, help="Camera focal length (pixels)")
    
    args = parser.parse_args()
    
    gen = SyntheticGenerator(
        width=args.width,
        height=args.height,
        background_dir=args.backgrounds,
        camera_pos=args.camera_pos,
        look_at=args.look_at,
        focal_length=args.focal_length
    )
    
    gen.run(args.num, args.output)
