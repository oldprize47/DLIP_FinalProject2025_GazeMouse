# 파일명: fginet.py

import torch
import torch.nn as nn
from stage1_gif_module import Stage1_GIFModule
from stage2_gif_module import Stage2_GIFModule
from stage3_gif_module import Stage3_GIFModule
from mlp_head import MLPHead


class FGINet(nn.Module):
    def __init__(self):
        super().__init__()
        # Stage 1, 2, 3 모듈 정의
        self.stage1 = Stage1_GIFModule(dropout_p=0.09)  # (B,96,56,56)
        self.stage2 = Stage2_GIFModule(dropout_p=0.06)  # (B,168,28,28)
        self.stage3 = Stage3_GIFModule(dropout_p=0.03)  # (B,336,14,14)

        # MLP Head: 화면상의 normalized X, Y 위치를 예측 → 2개 출력
        # 마지막에 Sigmoid를 적용해 출력이 [0,1] 범위로 제한합니다.
        self.mlp_head = nn.Sequential(MLPHead(in_features=336, hidden_dim=32, out_dim=2, dropout_p=0.1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224)
        x1 = self.stage1(x)  # → (B, 96, 56, 56)
        x2 = self.stage2(x)  # → (B,168, 28, 28)
        x3 = self.stage3(x)  # → (B,336, 14, 14)
        out = self.mlp_head(x3)  # → (B,2): [norm_x, norm_y]
        return out
