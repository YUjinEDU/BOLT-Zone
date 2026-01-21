"""
Squeeze-and-Excitation (SE) Attention Block

BOLT-Refine에서 블러/공 특징을 강조하기 위한 채널 어텐션 모듈.

BlurBall (2025) 논문에서 SE > CA > ECA 순으로 효과적임이 검증됨.
- SE: Global context를 활용한 채널별 재조정
- CA: 좌표 정보 포함 (작은 물체에는 노이즈 가능성)
- ECA: 경량화되었으나 short-range dependency만 학습

Reference:
- Hu et al. (2018): Squeeze-and-Excitation Networks (CVPR)
- BlurBall (2025): SE achieves best F1 for blur detection
"""

import torch
import torch.nn as nn
from typing import Optional


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block
    
    채널별 중요도를 학습하여 블러/공 특징 강조.
    Global Average Pooling → FC → ReLU → FC → Sigmoid 구조.
    
    Args:
        channels: 입력 채널 수
        reduction: FC 레이어의 차원 축소 비율 (default: 16)
    
    Example:
        >>> se = SEBlock(256, reduction=16)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> out = se(x)  # 동일 shape 출력
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        # Squeeze: Global Average Pooling
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        
        # Excitation: FC layers
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Output tensor of shape (B, C, H, W) with channel-wise recalibration
        """
        b, c, _, _ = x.size()
        
        # Squeeze: (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        y = self.squeeze(x).view(b, c)
        
        # Excitation: (B, C) -> (B, C)
        y = self.excitation(y).view(b, c, 1, 1)
        
        # Scale: element-wise multiplication
        return x * y.expand_as(x)


class CABlock(nn.Module):
    """Coordinate Attention Block
    
    위치 정보를 포함한 어텐션. 큰 물체에 효과적이나,
    작고 빠른 물체(야구공)에는 SE보다 효과가 떨어질 수 있음.
    
    Reference: Hou et al. (2021) - Coordinate Attention (CVPR)
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mid_channels = max(8, channels // reduction)
        
        self.conv1 = nn.Conv2d(channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(mid_channels, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, channels, 1, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        
        # Pool along H and W axes
        x_h = self.pool_h(x)  # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, W, 1)
        
        # Concatenate and conv
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)
        y = self.act(self.bn1(self.conv1(y)))
        
        # Split
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        # Attention
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        
        return x * a_h * a_w


class SEConvBlock(nn.Module):
    """SE Block이 적용된 Convolutional Block
    
    Conv → BN → ReLU → SE 구조로, BOLT-Refine의 Feature Extractor에 사용.
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, 
                 reduction: int = 16):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels, reduction)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.se(x)
        return x


if __name__ == "__main__":
    print("=== SE Attention Block Test ===")
    
    # SEBlock 테스트
    se = SEBlock(channels=256, reduction=16)
    x = torch.randn(2, 256, 32, 32)
    out = se(x)
    print(f"SEBlock: input {x.shape} -> output {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in se.parameters())}")
    
    # CABlock 테스트
    ca = CABlock(channels=256, reduction=16)
    out_ca = ca(x)
    print(f"CABlock: input {x.shape} -> output {out_ca.shape}")
    print(f"Parameters: {sum(p.numel() for p in ca.parameters())}")
    
    # SEConvBlock 테스트
    secb = SEConvBlock(in_channels=128, out_channels=256)
    x2 = torch.randn(2, 128, 64, 64)
    out2 = secb(x2)
    print(f"SEConvBlock: input {x2.shape} -> output {out2.shape}")
