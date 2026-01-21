# BOLT-Zone: Blur-aware Object Localization and Tracking for Strike Zone Judgment

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLO-26-green.svg)](https://docs.ultralytics.com/)
[![Hydra](https://img.shields.io/badge/Config-Hydra# ⚡ BOLT-Zone: Deep Learning Baseball Strike Zone

**BOLT-Zone**은 저사양 노트북(CPU)에서도 실시간으로 작동하는 딥러닝 기반 야구 스트라이크존 판정 시스템입니다.
기존의 색상 기반 추적이 아닌, **YOLO26**과 **Motion Blur Analysis**를 결합하여 다양한 조명과 환경에서도 강인한 성능을 보장합니다.

---

## 🚀 Key Features

### 1. Hybrid Detection Pipeline 🧠
- **BOLT-Detect (YOLO26n)**: 공의 위치를 빠르게 탐지 (Coarse Stage).
- **BOLT-Refine (YOLO26n-OBB)**: 모션 블러의 방향과 길이를 정밀 분석 (Fine Stage).
- **Adaptive Inference**: `GateNet`이 난이도를 판단하여 필요한 프레임만 정밀 분석 (Efficiency Up!).

### 2. Physics-Informed 3D Tracking ⚾
- **Aerodynamic Model**: 공기 저항($C_d$)과 중력($g$)을 고려한 물리 엔진 탑재.
- **Trajectory Fitting**: 노이즈가 섞인 관측 데이터에서도 실제 투구 궤적을 완벽하게 복원.

### 3. Quantitative Evaluation 📊
- **Benchmark Driven**: Recall 99%, Precision 95% 목표.
- **Latency Monitoring**: CPU p95 지연 시간 측정 시스템 내장.

---

## 📂 Project Structure

```bash
BOLT-Zone/
├── bolt/                # Core Python Package
│   ├── detect/          # YOLO26n Detector
│   ├── refine/          # YOLO26n-OBB Blur Analyzer
│   ├── track/           # Physics-based Tracker
│   │   └── physics.py   # ⚾ Physics Engine
│   ├── gate/            # Adaptive Inference
│   │   ├── engine.py    # Rule-based Engine
│   │   └── network.py   # 🧠 GateNet (MLP)
│   └── zone/            # Strike Zone Judgment
│
├── configs/             # Hydra Configurations
│   ├── model/           # Model Params
│   ├── dataset/         # Dataset & Augmentation
│   └── experimnet/      # Experiment Presets
│
├── data/                # Dataset Directory
│   ├── raw/             # YouTube Downloads
│   ├── yolo_detect/     # Detection Dataset
│   └── yolo_obb/        # OBB Dataset
│
├── docs/                # Documentation
│   ├── dataset_spec.md        # 📝 데이터셋 규격
│   ├── labeling_guide.md      # 🏷️ 라벨링 가이드 (OBB)
│   ├── evaluation_protocol.md # 📏 평가 프로토콜
│   └── youtube_download.md    # 📥 데이터 수집 가이드
│
└── scripts/             # Execution Scripts
    ├── train.py         # 학습 (Train/Val)
    ├── export.py        # 배포 (ONNX/OpenVINO)
    ├── evaluate.py      # 평가 (Metrics)
    ├── benchmark.py     # 성능 측정 (Latency)
    ├── train_gate.py    # GateNet 학습
    └── download_youtube.py # 데이터 수집
```

---

## ⚡ Quick Start

### 1. Installation

```bash
# Clone Repository
git clone https://github.com/yujin/BOLT-Zone.git
cd BOLT-Zone

# Install Dependencies
pip install -r requirements.txt
```

### 2. Data Collection

**데이터 수집 전략**: 실제 데이터 우선 + Augmentation

#### 2.1 YouTube 영상 다운로드 (Primary Source)

[YouTube 다운로드 가이드](docs/youtube_download_guide.md)를 참고하세요.

```bash
# 단일 영상 다운로드
python scripts/download_youtube.py --url "https://youtu.be/..." --domain umpire

# Manifest 기반 일괄 다운로드 (권장)
python scripts/download_youtube.py --manifest data/youtube_manifest.json
```

**추천 채널:**
- **심판 시점 (Umpire View)**: [MLB Official](https://www.youtube.com/@MLB), Skilled Catcher
- **포수 POV**: [POV BASEBALL](https://www.youtube.com/results?search_query=POV+BASEBALL)

#### 2.2 Data Augmentation

실제 데이터에 다양한 증강 기법을 적용하여 데이터셋 확장:
- 밝기/대비 조절 (야간/주간 시뮬레이션)
- 모션 블러 강도 조절
- 회전, Crop, Flip
- **Albumentations** 라이브러리 사용 (YOLO 학습 시 자동 적용)

> **Note**: 물리 기반 합성 데이터 생성(`scripts/generate_synthetic.py`)은 현재 보류 중입니다. 
> 실제 데이터와 괴리가 커서 학습 효과가 제한적이므로, 실제 데이터 수집 및 증강에 집중합니다.


### 3. Training

[학습 가이드](scripts/README_training.md)를 참고하세요.

```bash
# Detect 모델 학습
python scripts/train.py

# GateNet 학습 (Synthetic Data)
python scripts/train_gate.py
```

### 4. Benchmark

시스템의 실시간 성능을 측정합니다.

```bash
python scripts/benchmark.py --detect weights/best.onnx --refine weights/obb.onnx
```

---

## 📚 Documentation

- **[데이터셋 규격서](docs/dataset_spec.md)**: 데이터 포맷 및 물리 규격 정의
- **[라벨링 가이드](docs/labeling_guide.md)**: OBB 라벨링 방법론 (CVAT)
- **[평가 프로토콜](docs/evaluation_protocol.md)**: 성능 평가 지표 및 방법
- **[학습 가이드](scripts/README_training.md)**: 모델 학습부터 배포까지

---

## 🛠️ Tech Stack

- **Framework**: PyTorch, Ultralytics YOLO
- **Config**: Hydra, OmegaConf
- **Inference**: ONNX Runtime, OpenVINO
- **Ops**: WandB, TensorBoard

#### 실험 프리셋 예시
```bash
# 빠른 프로토타입 (작은 epoch, 작은 이미지)
python train.py +experiment=quick_prototype

# CPU 최적화 설정
python train.py +experiment=cpu_optimized
```

## 📚 문서

자세한 내용은 `docs/` 디렉토리를 참조하세요:
- `BOLT-Zone_v0.1.md`: 전체 시스템 설계 문서
- `dataset_spec.md`: 데이터셋 규격 (예정)
- `labeling_guide.md`: 라벨링 가이드 (예정)
- `evaluation_protocol.md`: 평가 프로토콜 (예정)

## 🔧 시스템 아키텍처

```
프레임 입력
    ↓
[BOLT-Detect] ← YOLO26n (빠른 bbox 검출)
    ↓
[BOLT-Track] ← ByteTrack/BoT-SORT (ID 부여)
    ↓
[BOLT-Gate] ← 불확실성 평가 → Refine ON/OFF
    ↓ (필요시만)
[BOLT-Refine] ← YOLO26n-OBB (블러 방향/길이)
    ↓
[BOLT-Zone] ← 궤적 계산 & Strike/Ball 판정
```

## 📊 평가 메트릭

- **Detection**: Recall, FP rate
- **Refine**: 중심 오차, 각도 오차, 길이 오차
- **End-to-End**: Strike/Ball 정확도, 교차점 오차
- **Real-time**: FPS, p95 지연, CPU 사용률

## 🤝 기여

이 프로젝트는 연구 목적으로 개발되었습니다.

## 📄 라이선스

TBD

## 🙏 감사

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - YOLO26, OBB, Track 모듈
- [Hydra](https://hydra.cc/) - 설정 관리 프레임워크
- AR_StrikeZone - 기존 ArUco 기반 스트라이크존 시스템
