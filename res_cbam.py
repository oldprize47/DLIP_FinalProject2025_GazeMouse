import math, os, random, time, warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from timm import create_model


# ───────────────────────────────────────────────────────────
# 1) Channel Attention
# ───────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        # MLP: in_channels → in_channels//reduction → in_channels
        self.mlp = nn.Sequential(nn.Linear(in_channels, in_channels // reduction, bias=True), nn.ReLU(inplace=True), nn.Linear(in_channels // reduction, in_channels, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # 1) Global AvgPool, MaxPool → (B, C, 1, 1) → view→ (B, C)
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(B, C)
        max_pool = F.adaptive_max_pool2d(x, 1).view(B, C)

        # 2) MLP 통과 → 두 결과 합산
        avg_out = self.mlp(avg_pool)  # (B, C)
        max_out = self.mlp(max_pool)  # (B, C)

        # 3) 합 → sigmoid → reshape to (B, C, 1, 1)
        attn = avg_out + max_out  # (B, C)
        attn = self.sigmoid(attn).view(B, C, 1, 1)  # (B, C, 1, 1)

        # 4) 입력 x에 채널별 곱 (broadcasting)
        return x * attn  # (B, C, H, W)


# ───────────────────────────────────────────────────────────
# 2) Spatial Attention
# ───────────────────────────────────────────────────────────
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size는 3 또는 7만 허용됩니다."
        padding = (kernel_size - 1) // 2

        # 입력 채널=2 → Conv → 출력 채널=1
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # 1) 채널축 평균/최대 → (B, 1, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)

        # 2) concat → (B, 2, H, W)
        concat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)

        # 3) Conv → (B, 1, H, W) → sigmoid
        attn = self.conv(concat)  # (B, 1, H, W)
        attn = self.sigmoid(attn)  # (B, 1, H, W)

        # 4) 입력 x에 공간별 곱
        return x * attn  # (B, C, H, W)


# ───────────────────────────────────────────────────────────
# 3) ResCBAM: Depthwise Separable Conv + CBAM + Residual
# ───────────────────────────────────────────────────────────
class ResCBAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        # 1) Depthwise Conv: groups=in_channels
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        # Pointwise Conv: 1×1
        self.pw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        # 2) CBAM 구성 요소
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        residual = x.clone()  # 나중에 더해줄 원본

        # -- Depthwise Conv → BN → ReLU --
        out = self.dw_conv(x)  # (B, C, H, W)
        out = self.bn(out)
        out = self.relu(out)

        # -- Pointwise Conv → BN → ReLU --
        out = self.pw_conv(out)  # (B, C, H, W)
        out = self.bn(out)
        out = self.relu(out)

        # -- Channel Attention --
        out = self.channel_att(out)  # (B, C, H, W)

        # -- Spatial Attention --
        out = self.spatial_att(out)  # (B, C, H, W)

        # -- Residual 연결 + ReLU --
        out = out + residual  # (B, C, H, W)
        out = self.relu(out)

        return out  # (B, C, H, W)


if __name__ == "__main__":
    dummy = torch.randn(2, 64, 32, 32)  # 예시: 배치 2, 채널 64, 크기 32×32
    model = ResCBAM(in_channels=64, reduction=16)
    out = model(dummy)
    print("Output shape:", out.shape)  # → 예상: torch.Size([2, 64, 32, 32]
