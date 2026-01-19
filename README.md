# BOLT-Zone: Blur-aware Object Localization and Tracking for Strike Zone Judgment

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLO-26-green.svg)](https://docs.ultralytics.com/)
[![Hydra](https://img.shields.io/badge/Config-Hydra-orange.svg)](https://hydra.cc/)

**BOLT-Zone**은 야구공의 **모션 블러를 정보로 활용**하는 blur-aware 철학을 기반으로, **YOLO26 + OBB(회전 박스)**와 **가변연산 게이팅**을 결합하여 **노트북 CPU에서도 실시간 스트라이크 판정**을 목표로 하는 시스템입니다.

## 🎯 핵심 특징

- **Blur-Aware Detection**: 모션 블러를 노이즈가 아닌 속도/방향 정보로 활용
- **2-Stage Architecture**: 가벼운 Detect + 정밀한 Refine (필요시만)
- **CPU Real-time**: 가변연산 게이팅으로 CPU에서도 실시간 동작
- **Experiment Management**: Hydra 기반 체계적인 실험 관리

## 📂 프로젝트 구조

```
BOLT-Zone/
├── bolt/                    # 핵심 모듈
│   ├── detect/              # YOLO26n 빠른 검출
│   ├── refine/              # YOLO26n-OBB 블러 정밀화
│   ├── track/               # ByteTrack/BoT-SORT 추적
│   ├── gate/                # 가변연산 게이팅 로직
│   ├── zone/                # 스트라이크존 판정
│   └── utils/               # 공통 유틸리티
│
├── configs/                 # Hydra 설정 파일
│   ├── config.yaml          # 메인 설정
│   ├── model/               # Detect/Refine 모델 설정
│   ├── dataset/             # 데이터셋 설정
│   ├── train/               # 학습 하이퍼파라미터
│   └── experiment/          # 실험별 프리셋
│
├── data/                    # 데이터셋
│   ├── raw/                 # 원본 영상
│   ├── clips/               # 공 등장 구간 클립
│   ├── yolo_detect/         # bbox 라벨
│   └── yolo_obb/            # OBB 라벨
│
├── scripts/                 # 스크립트
├── docs/                    # 문서
├── runs/                    # 학습 결과
└── outputs/                 # Hydra 실행 결과

```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd BOLT-Zone

# 의존성 설치
pip install -r requirements.txt
```

### 2. Hydra 기반 실험 실행

```bash
# 기본 학습 (Detect 모델)
python scripts/train.py

# 특정 실험 설정 사용
python scripts/train.py +experiment=quick_prototype

# 설정 오버라이드
python scripts/train.py model=refine train.epochs=50 device.type=cuda

# 멀티런 (하이퍼파라미터 스윕)
python scripts/train.py -m train.lr0=0.001,0.01,0.1
```

### 3. Hydra 설정 구조

#### 기본 사용법
```yaml
# configs/config.yaml이 기본 설정
# defaults로 각 카테고리별 설정 조합

defaults:
  - model: detect        # or refine
  - dataset: base        # or obb
  - train: default
  - experiment: null     # 선택적 오버라이드
```

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
