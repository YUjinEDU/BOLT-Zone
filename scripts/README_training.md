# BOLT-Zone 학습 파이프라인 빠른 시작

## 🎯 개요

데이터 라벨링 완료 후 학습부터 평가까지 전체 파이프라인을 실행하는 가이드입니다.

---

## 📦 1. 데이터 준비

### 1.1 데이터셋 구조

```
data/
├── yolo_detect/          # Detect 모델용
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── data.yaml         # YOLO 데이터셋 정의
│
└── yolo_obb/             # Refine 모델용 (OBB)
    ├── images/...
    ├── labels/...
    └── data.yaml
```

### 1.2 data.yaml 예시

`data/yolo_detect/data.yaml`:
```yaml
path: c:\Users\Yujin\My Drive\Workscpace\Projects\BOLT-Zone\data\yolo_detect
train: images/train
val: images/val
test: images/test

nc: 1
names: ['ball']
```

---

## 🚀 2. 학습 실행

### 2.1 Detect 모델 학습 (기본)

```bash
# 기본 학습
python scripts/train.py

# GPU 사용
python scripts/train.py device.type=cuda

# Epoch 조정
python scripts/train.py train.epochs=100
```

### 2.2 Refine 모델 학습 (OBB)

```bash
python scripts/train.py model=refine dataset=obb
```

### 2.3 빠른 프로토타입 (테스트용)

```bash
python scripts/train.py +experiment=quick_prototype
```

### 2.4 CPU 최적화 학습

```bash
python scripts/train.py +experiment=cpu_optimized
```

---

## 📤 3. 모델 Export

### 3.1 ONNX Export (범용)

```bash
python scripts/export.py --model runs/detect/best.pt
```

### 3.2 OpenVINO Export (CPU 최적화)

```bash
python scripts/export.py \
    --model runs/detect/best.pt \
    --format openvino
```

### 3.3 여러 형식 동시 Export

```bash
python scripts/export.py \
    --model runs/detect/best.pt \
    --format onnx openvino \
    --benchmark
```

---

## 📊 4. 평가 실행

### 4.1 기본 평가

```bash
python scripts/evaluate.py \
    --model runs/detect/best.pt \
    --data data/yolo_detect/data.yaml
```

### 4.2 실시간성 벤치마크 포함

```bash
python scripts/evaluate.py \
    --model runs/detect/best.pt \
    --data data/yolo_detect/data.yaml \
    --benchmark \
    --iterations 1000 \
    --plot
```

### 4.3 OBB 모델 평가

```bash
python scripts/evaluate.py \
    --model runs/refine/best.pt \
    --data data/yolo_obb/data.yaml \
    --task obb
```

### 4.4 상세 리포트 생성

```bash
python scripts/evaluate.py \
    --model runs/detect/best.pt \
    --data data/yolo_detect/data.yaml \
    --benchmark \
    --report results/eval_report.json \
    --plot
```

---

## 🎓 5. 전체 워크플로우 예시

### 5.1 Detect 모델 (처음부터 끝까지)

```bash
# 1. 학습
python scripts/train.py

# 2. Export
python scripts/export.py \
    --model runs/detect_bolt-zone-base/weights/best.pt \
    --format onnx openvino

# 3. 평가
python scripts/evaluate.py \
    --model runs/detect_bolt-zone-base/weights/best.pt \
    --data data/yolo_detect/data.yaml \
    --benchmark \
    --plot \
    --report results/detect_eval.json
```

### 5.2 하이퍼파라미터 스윕 (실험)

```bash
# Learning rate 스윕
python scripts/train.py -m train.lr0=0.001,0.01,0.1

# Batch size 스윕
python scripts/train.py -m train.batch=8,16,32
```

---

## 📈 6. 결과 확인

### 6.1 학습 결과

```
runs/
└── detect_bolt-zone-base/
    ├── weights/
    │   ├── best.pt        # 최고 성능 모델
    │   └── last.pt        # 마지막 epoch 모델
    ├── results.png        # 학습 곡선
    ├── confusion_matrix.png
    └── ...
```

### 6.2 평가 결과

```
results/
├── detect_eval.json       # 메트릭 JSON
└── latency_distribution.png  # 지연 히스토그램
```

### 6.3 Export 결과

```
runs/detect_bolt-zone-base/weights/
├── best.pt
├── best.onnx              # ONNX
└── best_openvino_model/   # OpenVINO
```

---

## 🎯 7. 목표 메트릭 체크리스트

학습 후 다음 목표값 달성 여부 확인:

### Detect 모델
- [ ] Recall ≥ 99% (공을 놓치지 않기)
- [ ] Precision ≥ 95%
- [ ] mAP@0.5 ≥ 97%

### 실시간성 (CPU)
- [ ] Mean latency < 50 ms
- [ ] p95 latency < 80 ms
- [ ] FPS ≥ 15 (ONNX) or ≥ 25 (OpenVINO)

---

## ⚠️ 문제 해결

### Q1: GPU out of memory

```bash
# Batch 크기 줄이기
python scripts/train.py train.batch=4
```

### Q2: 학습이 너무 느림 (CPU)

```bash
# 이미지 크기 줄이기
python scripts/train.py model.input.imgsz=320

# Workers 조정
python scripts/train.py device.num_workers=2
```

### Q3: Export 오류

```bash
# ONNX 먼저 시도
python scripts/export.py --model runs/detect/best.pt --format onnx

# OpenVINO는 별도 설치 필요
pip install openvino-dev
```

---

## 📚 다음 단계

1. **데이터 추가 수집** - 성능 향상
2. **BOLT-Gate 구현** - CPU 절약 (Phase 2B 계속)
3. **벤치마크** - p95 지연 측정
4. **논문 작성** - evaluation_protocol.md 참조

---

**참고 문서**:
- [dataset_spec.md](../docs/dataset_spec.md)
- [labeling_guide.md](../docs/labeling_guide.md)
- [evaluation_protocol.md](../docs/evaluation_protocol.md)
