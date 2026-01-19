"""
Hydra 기반 YOLO 학습 스크립트

Usage:
    # 기본 학습 (Detect)
    python train.py
    
    # Refine 모델 학습
    python train.py model=refine dataset=obb
    
    # 실험 프리셋 사용
    python train.py +experiment=quick_prototype
    
    # 설정 오버라이드
    python train.py train.epochs=100 train.batch=32
    
    # 멀티런 (하이퍼파라미터 스윕)
    python train.py -m train.lr0=0.001,0.01,0.1
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import logging

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Hydra 기반 YOLO 학습 메인 함수
    
    Args:
        cfg: Hydra DictConfig 객체 (configs/config.yaml 기준)
    """
    # 설정 출력 (디버깅용)
    log.info("=" * 60)
    log.info("BOLT-Zone Training Configuration")
    log.info("=" * 60)
    log.info(OmegaConf.to_yaml(cfg))
    
    # TODO: 실제 YOLO 학습 로직 구현
    # from ultralytics import YOLO
    # model = YOLO(cfg.model.weights.pretrained)
    # results = model.train(
    #     data=cfg.dataset.path,
    #     epochs=cfg.train.epochs,
    #     imgsz=cfg.model.input.imgsz,
    #     batch=cfg.train.batch,
    #     ...
    # )
    
    log.info("Training script placeholder - 실제 학습 로직은 추후 구현")
    log.info(f"Model: {cfg.model.name}")
    log.info(f"Dataset: {cfg.dataset.name}")
    log.info(f"Device: {cfg.device.type}")
    log.info(f"Epochs: {cfg.train.epochs}")


if __name__ == "__main__":
    main()
