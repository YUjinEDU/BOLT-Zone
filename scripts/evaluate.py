"""
BOLT-Zone Evaluation Script

학습된 모델을 evaluation_protocol.md에 정의된 메트릭으로 평가

Usage:
    # Detect 모델 평가
    python scripts/evaluate.py \
        --model runs/detect/best.pt \
        --data data/yolo_detect/data.yaml \
        --split test
    
    # Refine 모델 평가 (OBB)
    python scripts/evaluate.py \
        --model runs/refine/best.pt \
        --data data/yolo_obb/data.yaml \
        --task obb
    
    # 상세 리포트 생성
    python scripts/evaluate.py \
        --model runs/detect/best.pt \
        --data data/yolo_detect/data.yaml \
        --report results/eval_report.json \
        --plot
"""

import argparse
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Tuple
import logging
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


class BoltZoneEvaluator:
    """BOLT-Zone 평가 클래스"""
    
    def __init__(self, model_path: Path, data_yaml: Path, task: str = 'detect'):
        """
        Args:
            model_path: 모델 경로
            data_yaml: 데이터 YAML 경로
            task: 'detect' or 'obb'
        """
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.task = task
        
        # 모델 로드
        log.info(f"Loading model: {model_path}")
        self.model = YOLO(str(model_path))
        
        self.results = {}
    
    def evaluate_detect(self, split: str = 'test') -> Dict:
        """
        Detect 모델 평가 (evaluation_protocol.md Section 2)
        
        Returns:
            메트릭 딕셔너리
        """
        log.info("=" * 60)
        log.info("Evaluating Detect Model")
        log.info("=" * 60)
        
        # YOLO validation 실행
        val_results = self.model.val(
            data=str(self.data_yaml),
            split=split,
            save_json=True,
        )
        
        # 메트릭 추출
        metrics = {
            'detect': {
                'mAP@0.5': float(val_results.box.map50),
                'mAP@0.5:0.95': float(val_results.box.map),
                'precision': float(val_results.box.p),
                'recall': float(val_results.box.r),
                'f1_score': self._calc_f1(val_results.box.p, val_results.box.r),
            }
        }
        
        log.info(f"mAP@0.5: {metrics['detect']['mAP@0.5']:.4f}")
        log.info(f"Precision: {metrics['detect']['precision']:.4f}")
        log.info(f"Recall: {metrics['detect']['recall']:.4f}")
        log.info(f"F1-Score: {metrics['detect']['f1_score']:.4f}")
        
        # 목표값과 비교
        self._check_targets_detect(metrics['detect'])
        
        return metrics
    
    def evaluate_refine(self, split: str = 'test') -> Dict:
        """
        Refine 모델 평가 (OBB) (evaluation_protocol.md Section 3)
        
        Returns:
            메트릭 딕셔너리
        """
        log.info("=" * 60)
        log.info("Evaluating Refine Model (OBB)")
        log.info("=" * 60)
        
        # YOLO OBB validation
        val_results = self.model.val(
            data=str(self.data_yaml),
            split=split,
        )
        
        # OBB 메트릭
        metrics = {
            'refine': {
                'mAP@0.5': float(val_results.box.map50) if hasattr(val_results, 'box') else None,
                'obb_iou_mean': None,  # Ultralytics에서 직접 제공하지 않음 (커스텀 계산 필요)
                # 중심점/각도/길이 오차는 GT 데이터 필요 (별도 구현)
            }
        }
        
        log.info(f"mAP@0.5 (OBB): {metrics['refine']['mAP@0.5']:.4f}")
        log.warning("Center/Angle/Length errors require custom GT comparison")
        
        return metrics
    
    def evaluate_realtime(
        self,
        test_images: List[Path] = None,
        iterations: int = 100,
        device: str = 'cpu'
    ) -> Dict:
        """
        실시간성 평가 (evaluation_protocol.md Section 5)
        
        Args:
            test_images: 테스트 이미지 경로 리스트
            iterations: 반복 횟수
            device: 디바이스
        
        Returns:
            지연 통계
        """
        import time
        
        log.info("=" * 60)
        log.info("Evaluating Real-time Performance")
        log.info("=" * 60)
        
        # Dummy data 사용 (실제 이미지 없을 경우)
        if not test_images:
            log.warning("No test images provided. Using dummy input.")
            latencies = self._benchmark_dummy(iterations, device)
        else:
            latencies = self._benchmark_images(test_images[:iterations], device)
        
        # 통계 계산
        metrics = {
            'realtime': {
                'iterations': iterations,
                'device': device,
                'mean_latency_ms': float(np.mean(latencies)),
                'median_latency_ms': float(np.median(latencies)),
                'p95_latency_ms': float(np.percentile(latencies, 95)),
                'p99_latency_ms': float(np.percentile(latencies, 99)),
                'max_latency_ms': float(np.max(latencies)),
                'fps': float(1000 / np.mean(latencies)),
            }
        }
        
        log.info(f"Mean latency: {metrics['realtime']['mean_latency_ms']:.2f} ms")
        log.info(f"p95 latency: {metrics['realtime']['p95_latency_ms']:.2f} ms")
        log.info(f"FPS: {metrics['realtime']['fps']:.1f}")
        
        # 목표값과 비교
        self._check_targets_realtime(metrics['realtime'])
        
        return metrics, latencies
    
    def _benchmark_dummy(self, iterations: int, device: str) -> List[float]:
        """Dummy 입력으로 벤치마크"""
        import time
        
        latencies = []
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(10):
            _ = self.model.predict(dummy_img, device=device, verbose=False)
        
        for _ in range(iterations):
            start = time.perf_counter()
            _ = self.model.predict(dummy_img, device=device, verbose=False)
            latencies.append((time.perf_counter() - start) * 1000)
        
        return latencies
    
    def _benchmark_images(self, image_paths: List[Path], device: str) -> List[float]:
        """실제 이미지로 벤치마크"""
        import time
        from PIL import Image
        
        latencies = []
        
        for img_path in image_paths:
            img = Image.open(img_path)
            start = time.perf_counter()
            _ = self.model.predict(img, device=device, verbose=False)
            latencies.append((time.perf_counter() - start) * 1000)
        
        return latencies
    
    def _calc_f1(self, precision: float, recall: float) -> float:
        """F1 Score 계산"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def _check_targets_detect(self, metrics: Dict):
        """Detect 메트릭 목표값 체크"""
        targets = {
            'recall': 0.99,
            'precision': 0.95,
        }
        
        log.info("\nTarget Comparison:")
        for metric, target in targets.items():
            value = metrics.get(metric, 0)
            status = "✅" if value >= target else "❌"
            log.info(f"  {status} {metric}: {value:.4f} (target: {target:.2f})")
    
    def _check_targets_realtime(self, metrics: Dict):
        """실시간성 메트릭 목표값 체크"""
        if metrics['device'] == 'cpu':
            targets = {
                'mean_latency_ms': 50,
                'p95_latency_ms': 80,
            }
        else:
            targets = {}
        
        log.info("\nTarget Comparison:")
        for metric, target in targets.items():
            value = metrics.get(metric, 0)
            # 지연은 낮을수록 좋음
            status = "✅" if value <= target else "❌"
            log.info(f"  {status} {metric}: {value:.2f} ms (target: ≤{target} ms)")
    
    def plot_latency_distribution(
        self,
        latencies: List[float],
        output_path: Path = None
    ):
        """지연 분포 히스토그램"""
        plt.figure(figsize=(10, 6))
        
        plt.hist(latencies, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(latencies), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(latencies):.1f} ms')
        plt.axvline(np.percentile(latencies, 95), color='orange', linestyle='--',
                   label=f'p95: {np.percentile(latencies, 95):.1f} ms')
        
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency')
        plt.title('Inference Latency Distribution')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            log.info(f"Plot saved: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(self, output_path: Path):
        """평가 리포트 JSON 생성"""
        report = {
            'model': str(self.model_path),
            'task': self.task,
            'data': str(self.data_yaml),
            'metrics': self.results,
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        log.info(f"Report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate BOLT-Zone models'
    )
    parser.add_argument(
        '--model',
        type=Path,
        required=True,
        help='Path to model (.pt file)'
    )
    parser.add_argument(
        '--data',
        type=Path,
        required=True,
        help='Path to data.yaml'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='detect',
        choices=['detect', 'obb'],
        help='Task type'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='Dataset split to evaluate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device (cpu, cuda, etc.)'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run real-time performance benchmark'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Benchmark iterations'
    )
    parser.add_argument(
        '--report',
        type=Path,
        help='Output report JSON path'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate plots'
    )
    
    args = parser.parse_args()
    
    # Evaluator 생성
    evaluator = BoltZoneEvaluator(args.model, args.data, args.task)
    
    # 평가 실행
    if args.task == 'detect':
        metrics = evaluator.evaluate_detect(args.split)
        evaluator.results.update(metrics)
    elif args.task == 'obb':
        metrics = evaluator.evaluate_refine(args.split)
        evaluator.results.update(metrics)
    
    # 실시간성 평가
    if args.benchmark:
        rt_metrics, latencies = evaluator.evaluate_realtime(
            iterations=args.iterations,
            device=args.device
        )
        evaluator.results.update(rt_metrics)
        
        # 히스토그램
        if args.plot:
            plot_path = args.model.parent / 'latency_distribution.png'
            evaluator.plot_latency_distribution(latencies, plot_path)
    
    # 리포트 생성
    if args.report:
        evaluator.generate_report(args.report)
    
    log.info("=" * 60)
    log.info("Evaluation Complete!")
    log.info("=" * 60)


if __name__ == '__main__':
    main()
