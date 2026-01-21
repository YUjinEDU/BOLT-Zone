"""
BOLT-Zone System Benchmark

CPU 환경에서 End-to-End 파이프라인(Detect + Gate + Refine)의 지연 시간을 측정합니다.
p95, p99 지연 시간을 통해 실시간성을 검증합니다.

Usage:
    python scripts/benchmark.py --detect weights/detect.onnx --refine weights/refine.onnx
"""

import argparse
import time
import numpy as np
from pathlib import Path
import logging
import json

# ONNX Runtime
import onnxruntime as ort

# BOLT Modules
from bolt.gate.engine import BoltGate

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class SystemBenchmark:
    def __init__(self, detect_path: str, refine_path: str, device: str = 'cpu'):
        self.device = device
        providers = ['CPUExecutionProvider']
        if device == 'cuda':
            providers.insert(0, 'CUDAExecutionProvider')
            
        log.info(f"Loading Detect Model: {detect_path}")
        self.sess_detect = ort.InferenceSession(detect_path, providers=providers)
        
        log.info(f"Loading Refine Model: {refine_path}")
        self.sess_refine = ort.InferenceSession(refine_path, providers=providers)
        
        self.gate = BoltGate(conf_threshold=0.6)
        
        # Input names
        self.in_name_det = self.sess_detect.get_inputs()[0].name
        self.in_name_ref = self.sess_refine.get_inputs()[0].name
        
        # Prepare Dummy Inputs
        self.dummy_img = np.random.randn(1, 3, 640, 640).astype(np.float32)
        
        # Warmup
        log.info("Warming up...")
        for _ in range(10):
            self.sess_detect.run(None, {self.in_name_det: self.dummy_img})
            self.sess_refine.run(None, {self.in_name_ref: self.dummy_img})
            
    def run(self, iterations: int = 1000, gating_ratio: float = 0.5):
        """
        벤치마크 실행
        
        Args:
            iterations: 반복 횟수
            gating_ratio: 시뮬레이션할 Refine 비율 (0.0 ~ 1.0)
                          (실제 Gate 로직 대신 확률적으로 Refine 수행)
        """
        latencies = []
        gate_decisions = []
        
        log.info(f"Starting Benchmark: {iterations} iterations, Target Refine Ratio: {gating_ratio}")
        
        start_time = time.perf_counter()
        
        for i in range(iterations):
            iter_start = time.perf_counter()
            
            # 1. Detect Inference
            det_out = self.sess_detect.run(None, {self.in_name_det: self.dummy_img})
            
            # 2. Gate Decision (Simulation)
            # 실제로는 det_out을 파싱해야 하지만, 벤치마크에서는 비율 맞춰 시뮬레이션
            # 랜덤하게 Refine 결정
            need_refine = np.random.random() < gating_ratio
            gate_decisions.append(need_refine)
            
            # 3. Refine Inference (Conditional)
            if need_refine:
                # ROI crop 과정 생략 (순수 추론 시간 측정)
                ref_out = self.sess_refine.run(None, {self.in_name_ref: self.dummy_img})
            
            iter_end = time.perf_counter()
            latencies.append((iter_end - iter_start) * 1000)  # ms
            
        total_time = time.perf_counter() - start_time
        
        self.print_stats(latencies, gate_decisions)
        
        return latencies
        
    def print_stats(self, latencies: list, gate_decisions: list):
        latencies = np.array(latencies)
        refine_count = sum(gate_decisions)
        total = len(latencies)
        actual_ratio = refine_count / total
        
        log.info("=" * 60)
        log.info("BENCHMARK RESULTS")
        log.info("=" * 60)
        log.info(f"Total Iterations: {total}")
        log.info(f"Refine Ratio: {actual_ratio:.2%} (Count: {refine_count})")
        log.info("-" * 30)
        log.info(f"Mean Latency:   {np.mean(latencies):.2f} ms")
        log.info(f"Median Latency: {np.median(latencies):.2f} ms")
        log.info(f"p95 Latency:    {np.percentile(latencies, 95):.2f} ms")
        log.info(f"p99 Latency:    {np.percentile(latencies, 99):.2f} ms")
        log.info(f"Max Latency:    {np.max(latencies):.2f} ms")
        log.info("-" * 30)
        log.info(f"Estimated FPS:  {1000 / np.mean(latencies):.1f} FPS")
        log.info("=" * 60)
        
        # Target Check (Laptop CPU Target: 30 FPS / < 33ms)
        p95 = np.percentile(latencies, 95)
        if p95 < 33.3:
            log.info("✅ Perfect Real-time (30 FPS+)")
        elif p95 < 66.6:
            log.info("⚠️ Playable (15 FPS+)")
        else:
            log.info("❌ Too Slow (< 15 FPS)")


def main():
    parser = argparse.ArgumentParser(description='BOLT-Zone System Benchmark')
    parser.add_argument('--detect', type=str, required=True, help='Path to Detect ONNX model')
    parser.add_argument('--refine', type=str, required=True, help='Path to Refine ONNX model')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--iters', type=int, default=1000, help='Iterations')
    parser.add_argument('--ratio', type=float, default=0.3, help='Simulated Refine Ratio (0.0-1.0)')
    
    args = parser.parse_args()
    
    if not Path(args.detect).exists() or not Path(args.refine).exists():
        log.error("Model files not found!")
        return
        
    bench = SystemBenchmark(args.detect, args.refine, args.device)
    bench.run(args.iters, args.ratio)


if __name__ == '__main__':
    main()
