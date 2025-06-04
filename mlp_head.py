# 파일명: mlp_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPHead(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 32, out_dim: int = 2, dropout_p: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True), nn.Dropout(p=dropout_p), nn.Linear(hidden_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → Global AvgPool → (B, C)
        B, C, H, W = x.shape
        x = F.adaptive_avg_pool2d(x, 1).view(B, C)  # (B, C)
        out = self.fc(x)  # (B, 2)
        return out
