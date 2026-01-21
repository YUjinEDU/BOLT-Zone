"""
Train GateNet

GateNet 학습 스크립트.
Detect 결과와 GT(Refine이 필요한지 여부)를 기반으로 학습합니다.
데이터셋은 시뮬레이션으로 생성하거나 실제 라벨링 데이터로부터 추출할 수 있습니다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from pathlib import Path
import logging

from bolt.gate.network import GateNet

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class GateDataset(Dataset):
    def __init__(self, data_path: Path = None, num_samples: int = 1000):
        """
        Args:
            data_path: 실제 데이터 경로 (없으면 시뮬레이션 데이터 생성)
            num_samples: 시뮬레이션 샘플 수
        """
        if data_path and data_path.exists():
            # Load real data (Not implemented yet)
            raise NotImplementedError("Real data loading not implemented yet")
        else:
            # Generate Synthetic Data for prototyping
            log.info(f"Generating {num_samples} synthetic samples...")
            self.X, self.y = self._generate_synthetic(num_samples)
            
    def _generate_synthetic(self, n: int):
        # Feature: [conf, w, h, vx, vy, uncertainty]
        X = np.random.rand(n, 6).astype(np.float32)
        y = np.zeros(n, dtype=np.float32)
        
        # Rule-based Logic을 모사하여 GT 생성 (노이즈 추가)
        for i in range(n):
            conf = X[i, 0]
            size = min(X[i, 1], X[i, 2])
            uncertainty = X[i, 5]
            
            # Refine 조건 (가정)
            # 1. Conf가 낮으면 Refine (0.6 이하)
            # 2. 크기가 작으면 Refine (0.1 이하)
            # 3. 불확실성이 크면 Refine (0.8 이상)
            
            score = 0
            if conf < 0.6: score += 1
            if size < 0.1: score += 1
            if uncertainty > 0.8: score += 1
            
            # 노이즈를 섞어서 "애매한" 케이스 생성
            label = 1.0 if score >= 1 else 0.0
            
            # Random Noise flip (5%)
            if np.random.rand() < 0.05:
                label = 1.0 - label
                
            y[i] = label
            
        return torch.tensor(X), torch.tensor(y).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GateNet(input_dim=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    
    # Data
    dataset = GateDataset(num_samples=args.samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Train Loop
    log.info(f"Start training on {device}")
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (pred > 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)
            
        acc = correct / total
        if (epoch + 1) % 10 == 0:
            log.info(f"Epoch {epoch+1}/{args.epochs} - Loss: {total_loss/len(loader):.4f} - Acc: {acc:.4f}")
            
    # Save
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), save_dir / "gatenet.pt")
    log.info(f"Model saved to {save_dir / 'gatenet.pt'}")
    
    # Export to ONNX
    from bolt.gate.network import export_gate_model
    export_gate_model(str(save_dir / "gatenet.pt"), input_dim=6)


def main():
    parser = argparse.ArgumentParser(description="Train GateNet")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--samples", type=int, default=1000, help="Number of synthetic samples")
    parser.add_argument("--output_dir", type=str, default="weights", help="Output directory")
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
