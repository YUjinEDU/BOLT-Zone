"""
BOLT-Zone Model Export Script

학습된 YOLO 모델을 다양한 형식으로 Export (ONNX, OpenVINO, TensorRT 등)

Usage:
    # ONNX Export (기본)
    python scripts/export.py --model runs/detect/best.pt
    
    # OpenVINO Export (CPU 최적화)
    python scripts/export.py --model runs/detect/best.pt --format openvino
    
    # 여러 형식으로 한번에
    python scripts/export.py --model runs/detect/best.pt --format onnx openvino
    
    # 커스텀 출력 디렉토리
    python scripts/export.py --model runs/detect/best.pt --output exports/
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


SUPPORTED_FORMATS = {
    'onnx': {
        'extension': '.onnx',
        'description': 'ONNX (Open Neural Network Exchange) - 범용',
        'args': {
            'simplify': True,
            'dynamic': False,
        }
    },
    'openvino': {
        'extension': '_openvino_model',
        'description': 'OpenVINO - Intel CPU/GPU 최적화',
        'args': {
            'half': False,  # CPU에서는 FP16 비활성화
        }
    },
    'tensorrt': {
        'extension': '.engine',
        'description': 'TensorRT - NVIDIA GPU 최적화',
        'args': {
            'half': True,
            'workspace': 4,  # GB
        }
    },
    'coreml': {
        'extension': '.mlmodel',
        'description': 'CoreML - Apple Silicon',
        'args': {}
    },
    'torchscript': {
        'extension': '.torchscript',
        'description': 'TorchScript - PyTorch 최적화',
        'args': {}
    },
}


def export_model(
    model_path: Path,
    format: str,
    output_dir: Path = None,
    imgsz: int = 640,
    half: bool = False,
    **kwargs
) -> Path:
    """
    YOLO 모델을 지정된 형식으로 Export
    
    Args:
        model_path: 학습된 모델 경로 (.pt)
        format: Export 형식 (onnx, openvino, tensorrt 등)
        output_dir: 출력 디렉토리 (None이면 모델 디렉토리)
        imgsz: 입력 이미지 크기
        half: FP16 사용 여부
        **kwargs: 추가 Export 옵션
    
    Returns:
        Export된 모델 경로
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if format not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {format}. "
            f"Supported: {list(SUPPORTED_FORMATS.keys())}"
        )
    
    log.info("=" * 60)
    log.info(f"Exporting Model: {model_path.name}")
    log.info(f"Format: {format} - {SUPPORTED_FORMATS[format]['description']}")
    log.info("=" * 60)
    
    # 모델 로드
    model = YOLO(str(model_path))
    
    # Export 파라미터 구성
    export_args = {
        'format': format,
        'imgsz': imgsz,
        'half': half,
        **SUPPORTED_FORMATS[format]['args'],
        **kwargs,
    }
    
    log.info(f"Export arguments: {export_args}")
    
    # Export 실행
    try:
        export_path = model.export(**export_args)
        log.info(f"✅ Export successful: {export_path}")
        
        # 출력 디렉토리로 이동 (선택)
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            export_path_obj = Path(export_path)
            
            if export_path_obj.is_file():
                # 단일 파일
                dest = output_dir / export_path_obj.name
                import shutil
                shutil.copy2(export_path, dest)
                log.info(f"Copied to: {dest}")
                return dest
            elif export_path_obj.is_dir():
                # 디렉토리 (OpenVINO 등)
                dest = output_dir / export_path_obj.name
                import shutil
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(export_path, dest)
                log.info(f"Copied to: {dest}")
                return dest
        
        return Path(export_path)
    
    except Exception as e:
        log.error(f"❌ Export failed: {e}")
        raise


def benchmark_export(export_path: Path, device: str = 'cpu', iterations: int = 100):
    """
    Export된 모델의 추론 속도 벤치마크
    
    Args:
        export_path: Export된 모델 경로
        device: 디바이스 (cpu, cuda)
        iterations: 반복 횟수
    """
    import time
    import numpy as np
    
    log.info("=" * 60)
    log.info(f"Benchmarking: {export_path}")
    log.info("=" * 60)
    
    # ONNX Runtime 예시
    if str(export_path).endswith('.onnx'):
        try:
            import onnxruntime as ort
            
            # Session 생성
            providers = ['CPUExecutionProvider']
            if device == 'cuda':
                providers.insert(0, 'CUDAExecutionProvider')
            
            session = ort.InferenceSession(str(export_path), providers=providers)
            
            # Dummy input
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            # Dynamic batch를 1로 설정
            if isinstance(input_shape[0], str):
                input_shape = [1] + list(input_shape[1:])
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
            # Warmup
            for _ in range(10):
                _ = session.run(None, {input_name: dummy_input})
            
            # Benchmark
            latencies = []
            for _ in range(iterations):
                start = time.perf_counter()
                _ = session.run(None, {input_name: dummy_input})
                latencies.append((time.perf_counter() - start) * 1000)  # ms
            
            log.info(f"Iterations: {iterations}")
            log.info(f"Mean latency: {np.mean(latencies):.2f} ms")
            log.info(f"Median latency: {np.median(latencies):.2f} ms")
            log.info(f"p95 latency: {np.percentile(latencies, 95):.2f} ms")
            log.info(f"p99 latency: {np.percentile(latencies, 99):.2f} ms")
            log.info(f"FPS: {1000 / np.mean(latencies):.1f}")
            
        except ImportError:
            log.warning("onnxruntime not installed. Skipping benchmark.")
    else:
        log.info("Benchmark only supported for ONNX format currently.")


def main():
    parser = argparse.ArgumentParser(
        description='Export YOLO models to various formats for deployment'
    )
    parser.add_argument(
        '--model',
        type=Path,
        required=True,
        help='Path to trained model (.pt file)'
    )
    parser.add_argument(
        '--format',
        type=str,
        nargs='+',
        default=['onnx'],
        choices=list(SUPPORTED_FORMATS.keys()),
        help='Export format(s)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output directory (default: same as model)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size (default: 640)'
    )
    parser.add_argument(
        '--half',
        action='store_true',
        help='Use FP16 precision (GPU only)'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run inference benchmark after export'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device for benchmark (default: cpu)'
    )
    
    args = parser.parse_args()
    
    # 모델 존재 확인
    if not args.model.exists():
        log.error(f"Model not found: {args.model}")
        sys.exit(1)
    
    # Format이 리스트인지 확인
    formats = args.format if isinstance(args.format, list) else [args.format]
    
    # 각 format별 Export
    exported_paths = []
    for fmt in formats:
        try:
            export_path = export_model(
                model_path=args.model,
                format=fmt,
                output_dir=args.output,
                imgsz=args.imgsz,
                half=args.half,
            )
            exported_paths.append((fmt, export_path))
            
            # Benchmark (ONNX만)
            if args.benchmark and fmt == 'onnx':
                benchmark_export(export_path, args.device)
        
        except Exception as e:
            log.error(f"Failed to export {fmt}: {e}")
            continue
    
    # 요약
    log.info("=" * 60)
    log.info("Export Summary")
    log.info("=" * 60)
    for fmt, path in exported_paths:
        log.info(f"✅ {fmt.upper()}: {path}")
    
    if not exported_paths:
        log.error("❌ No models were exported successfully")
        sys.exit(1)


if __name__ == '__main__':
    main()
