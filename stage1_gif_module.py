# 파일명: stage1_gif_module.py

import torch
import torch.nn as nn
from timm import create_model
from res_cbam import ResCBAM  # res_cbam.py 파일과 동일 폴더에 있어야 합니다.


class Stage1_GIFModule(nn.Module):
    def __init__(self, dropout_p: float = 0.1):
        super().__init__()

        # ─────────────────────────────────────────────────
        # 1) EfficientNet-B0: (B, 3, 224, 224) → feat_list[1] = (B, 24, 56, 56)
        # ─────────────────────────────────────────────────
        self.efficient_feats = create_model(
            "efficientnet_b0",
            pretrained=False,
            features_only=True,
            out_indices=(1,),  # EfficientNet stage1 feature (56×56)
        )
        # EfficientNet 에서 뽑힌 채널 수 (예: 24)
        self.eff_out_channels = self.efficient_feats.feature_info.channels()[0]

        # ─────────────────────────────────────────────────
        # 2) Swin Tiny: (B, 3, 224, 224) → feat_list[0] = (B, 56, 56, 96)
        # ─────────────────────────────────────────────────
        self.swin_feats = create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            features_only=True,
            out_indices=(0,),  # SwinTiny stage0 feature (56×56)
        )
        # SwinTiny에서는 (B, 56, 56, 96) 형태로 나옴 → permute 필요
        # 채널 수는 permute 후 두 번째 차원(96)
        self.swin_out_channels = self.swin_feats.feature_info.channels()[0]

        # ─────────────────────────────────────────────────
        # 3) Concat 후 1×1 Conv로 채널 줄이기
        #    fused_channels = eff_out_channels + swin_out_channels = 24 + 96 = 120
        #    target_channels = 96 (논문 예시)
        # ─────────────────────────────────────────────────
        fused_channels = self.eff_out_channels + self.swin_out_channels  # 24 + 96 = 120
        target_channels = 96

        self.reduce_conv = nn.Conv2d(in_channels=fused_channels, out_channels=target_channels, kernel_size=1, bias=False)
        self.reduce_bn = nn.BatchNorm2d(target_channels)
        self.reduce_relu = nn.ReLU(inplace=True)

        # ─────────────────────────────────────────────────
        # 4) ResCBAM 모듈 (in_channels = target_channels, reduction=16)
        # ─────────────────────────────────────────────────
        self.res_cbam = ResCBAM(in_channels=target_channels, reduction=16)

        # ─────────────────────────────────────────────────
        # 5) Dropout2d (p=0.1)
        # ─────────────────────────────────────────────────
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224)

        # 1) EfficientNet Stage1 feature 추출 → (B, 24, 56, 56)
        eff_feats = self.efficient_feats(x)[0]

        # 2) SwinTiny Stage0 feature 추출 → (B, 56, 56, 96) → permute → (B, 96, 56, 56)
        swin_feats = self.swin_feats(x)[0]  # (B, 56, 56, 96)
        swin_feats = swin_feats.permute(0, 3, 1, 2)  # → (B, 96, 56, 56)

        # 3) Concatenate (채널 방향) → (B, 120, 56, 56)
        fused = torch.cat([eff_feats, swin_feats], dim=1)  # (24+96=120 채널)

        # 4) 1×1 Conv + BN + ReLU → (B, 96, 56, 56)
        out = self.reduce_conv(fused)
        out = self.reduce_bn(out)
        out = self.reduce_relu(out)

        # 5) ResCBAM → (B, 96, 56, 56)
        out = self.res_cbam(out)

        # 6) Dropout2d → (B, 96, 56, 56)
        out = self.dropout(out)

        return out  # (B, 96, 56, 56)
