"""
BOLT-Zone Training Script (Ultralytics YOLO Integration)

Hydra 기반 실험 관리와 Ultralytics YOLO를 통합한 학습 스크립트

Usage:
    # Detect 모델 학습 (기본)
    python scripts/train.py
    
    # Refine 모델 학습 (OBB)
    python scripts/train.py model=refine dataset=obb
    
    # 실험 프리셋 사용
    python scripts/train.py +experiment=quick_prototype
    
    # 하이퍼파라미터 오버라이드
    python scripts/train.py train.epochs=100 train.batch=32 device.type=cuda
    
    # 멀티런 (하이퍼파라미터 스윕)
    python scripts/train.py -m train.lr0=0.001,0.01,0.1
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import logging
from typing import Optional
import torch
from ultralytics import YOLO

log = logging.getLogger(__name__)


def setup_directories(cfg: DictConfig) -> dict:
    """학습 디렉토리 구조 생성"""
    dirs = {
        'data': Path(cfg.project.data_dir),
        'output': Path(cfg.project.output_dir),
        'runs': Path(cfg.project.root_dir) / 'runs',
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def prepare_data_yaml(cfg: DictConfig, dirs: dict) -> Path:
    """
    YOLO data.yaml 파일 준비
    
    Hydra 설정을 YOLO 형식으로 변환
    """
    import yaml
    
    # Dataset 설정에서 YOLO data.yaml 정보 추출
    data_yaml = {
        'path': cfg.dataset.path,
        'train': cfg.dataset.train,
        'val': cfg.dataset.val,
        'test': cfg.dataset.get('test', None),
        'nc': cfg.dataset.nc,
        'names': cfg.dataset.names,
    }
    
    # 임시 data.yaml 파일 생성 (Hydra output 디렉토리)
    data_yaml_path = Path.cwd() / 'data.yaml'
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
    
    log.info(f"Created data.yaml: {data_yaml_path}")
    return data_yaml_path


def create_model(cfg: DictConfig) -> YOLO:
    """
    YOLO 모델 생성 또는 로드
    
    Args:
        cfg: Hydra config
    
    Returns:
        YOLO model instance
    """
    # Checkpoint 우선 (resume 시)
    if cfg.train.resume and cfg.model.weights.checkpoint:
        checkpoint_path = Path(cfg.model.weights.checkpoint)
        if checkpoint_path.exists():
            log.info(f"Resuming from checkpoint: {checkpoint_path}")
            return YOLO(str(checkpoint_path))
    
    # Pretrained weights 사용
    pretrained = cfg.model.weights.pretrained
    log.info(f"Loading pretrained model: {pretrained}")
    
    try:
        model = YOLO(pretrained)
        return model
    except Exception as e:
        log.error(f"Failed to load model {pretrained}: {e}")
        raise


def train_model(model: YOLO, cfg: DictConfig, data_yaml_path: Path) -> dict:
    """
    YOLO 모델 학습
    
    Args:
        model: YOLO model
        cfg: Hydra config
        data_yaml_path: Path to data.yaml
    
    Returns:
        Training results dict
    """
    log.info("=" * 60)
    log.info("Starting Training")
    log.info("=" * 60)
    
    # Ultralytics 학습 파라미터 구성
    train_args = {
        # Data
        'data': str(data_yaml_path),
        
        # Training
        'epochs': cfg.train.epochs,
        'batch': cfg.train.batch,
        'imgsz': cfg.model.input.imgsz,
        
        # Optimizer
        'optimizer': cfg.train.optimizer,
        'lr0': cfg.train.lr0,
        'lrf': cfg.train.lrf,
        'momentum': cfg.train.momentum,
        'weight_decay': cfg.train.weight_decay,
        
        # Scheduler
        'cos_lr': cfg.train.scheduler == 'cosine',
        
        # Loss
        'box': cfg.train.box,
        'cls': cfg.train.cls,
        'dfl': cfg.train.dfl,
        
        # Validation
        'val': cfg.train.val,
        
        # Device
        'device': cfg.device.type,
        'workers': cfg.device.num_workers,
        
        # AMP
        'amp': cfg.train.amp,
        
        # Augmentation (dataset에서 가져옴)
        'hsv_h': cfg.dataset.augmentation.hsv_h,
        'hsv_s': cfg.dataset.augmentation.hsv_s,
        'hsv_v': cfg.dataset.augmentation.hsv_v,
        'degrees': cfg.dataset.augmentation.degrees,
        'translate': cfg.dataset.augmentation.translate,
        'scale': cfg.dataset.augmentation.scale,
        'shear': cfg.dataset.augmentation.shear,
        'perspective': cfg.dataset.augmentation.perspective,
        'flipud': cfg.dataset.augmentation.flipud,
        'fliplr': cfg.dataset.augmentation.fliplr,
        'mosaic': cfg.dataset.augmentation.mosaic,
        'mixup': cfg.dataset.augmentation.mixup,
        
        # 기타
        'patience': cfg.train.patience,
        'save': cfg.train.save.best,
        'save_period': cfg.train.save_period,
        'project': 'runs',
        'name': f"{cfg.model.type}_{cfg.dataset.name}",
        'exist_ok': True,
        
        # Wandb (선택)
        'wandb': cfg.logging.wandb.enabled,
    }
    
    log.info("Training arguments:")
    log.info(OmegaConf.to_yaml(train_args))
    
    # 학습 실행
    try:
        results = model.train(**train_args)
        log.info("Training completed successfully!")
        return results
    except Exception as e:
        log.error(f"Training failed: {e}")
        raise


def validate_model(model: YOLO, cfg: DictConfig, data_yaml_path: Path) -> dict:
    """
    학습된 모델 검증
    
    Args:
        model: Trained YOLO model
        cfg: Hydra config
        data_yaml_path: Path to data.yaml
    
    Returns:
        Validation results dict
    """
    log.info("=" * 60)
    log.info("Running Validation")
    log.info("=" * 60)
    
    val_args = {
        'data': str(data_yaml_path),
        'split': 'test',  # Test split 사용
        'batch': cfg.train.batch,
        'imgsz': cfg.model.input.imgsz,
        'device': cfg.device.type,
    }
    
    try:
        results = model.val(**val_args)
        
        # 주요 메트릭 출력
        if hasattr(results, 'box'):
            log.info(f"mAP@0.5: {results.box.map50:.4f}")
            log.info(f"mAP@0.5:0.95: {results.box.map:.4f}")
            log.info(f"Precision: {results.box.p:.4f}")
            log.info(f"Recall: {results.box.r:.4f}")
        
        return results
    except Exception as e:
        log.error(f"Validation failed: {e}")
        raise


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Hydra 기반 YOLO 학습 메인 함수
    
    Args:
        cfg: Hydra DictConfig 객체
    """
    # 시드 설정 (재현성)
    if cfg.seed:
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        log.info(f"Random seed set: {cfg.seed}")
    
    # 설정 출력
    log.info("=" * 60)
    log.info("BOLT-Zone Training Configuration")
    log.info("=" * 60)
    log.info(f"\nModel: {cfg.model.name}")
    log.info(f"Dataset: {cfg.dataset.name}")
    log.info(f"Task: {cfg.model.task}")
    log.info(f"Device: {cfg.device.type}")
    log.info(f"\nFull Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # 디렉토리 설정
    dirs = setup_directories(cfg)
    
    # Data YAML 준비
    data_yaml_path = prepare_data_yaml(cfg, dirs)
    
    # 모드별 실행
    if cfg.mode == 'train':
        # 모델 생성
        model = create_model(cfg)
        
        # 학습 실행
        results = train_model(model, cfg, data_yaml_path)
        
        # 학습 완료 후 검증 (선택)
        if cfg.train.val:
            # Best 모델 로드
            best_model_path = Path('runs') / f"{cfg.model.type}_{cfg.dataset.name}" / 'weights' / 'best.pt'
            if best_model_path.exists():
                log.info(f"Loading best model: {best_model_path}")
                best_model = YOLO(str(best_model_path))
                validate_model(best_model, cfg, data_yaml_path)
        
        log.info(f"Training outputs saved to: runs/{cfg.model.type}_{cfg.dataset.name}")
    
    elif cfg.mode == 'eval':
        # 평가만 수행
        checkpoint = cfg.model.weights.checkpoint
        if not checkpoint:
            raise ValueError("Checkpoint path required for eval mode")
        
        model = YOLO(checkpoint)
        validate_model(model, cfg, data_yaml_path)
    
    else:
        log.error(f"Unknown mode: {cfg.mode}")
        raise ValueError(f"Mode must be 'train' or 'eval', got '{cfg.mode}'")


if __name__ == "__main__":
    main()
