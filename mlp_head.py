# 파일명: mlp_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPHead(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 32, out_dim: int = 2, dropout_p: float = 0.2):
        super().__init__()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(in_features, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True), nn.Dropout(p=dropout_p), nn.Linear(hidden_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x가 4-D이면 (B,C,H,W) → GAP, 2-D이면 (B,C) 그대로 FC.
        """
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = F.adaptive_avg_pool2d(x, 1).view(B, C)
        # else: 이미 (B,C)
        return self.fc(x)  # (B,2)
