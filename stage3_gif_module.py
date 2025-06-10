# 파일명: stage3_gif_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from res_cbam import ResCBAM  # 앞서 만든 res_cbam.py 파일이 동일 디렉터리에 있어야 합니다.


class Stage3_GIFModule(nn.Module):
    def __init__(self, dropout_p: float = 0.03):
        super().__init__()

        # ─────────────────────────────────────────────────
        # 1) EfficientNet-B0 Stage3 설정
        #    - 원본 이미지 (B,3,224,224) → feat_list[3] = (B, C_eff3, 14, 14) 예상
        # ─────────────────────────────────────────────────
        self.efficient_feats = create_model(
            "efficientnet_b0",
            pretrained=True,
            features_only=True,
            out_indices=(3,),  # EfficientNet stage3 feature (14×14)
        )
        # Stage3에서 뽑히는 채널 수 (예: 40 또는 80 등)
        self.eff3_out_channels = self.efficient_feats.feature_info.channels()[0]
        # 실제로 나온 값을 확인해주세요.
        # (예시: self.eff3_out_channels == 40)

        # ─────────────────────────────────────────────────
        # 2) Swin Tiny Stage2 설정
        #    - 원본 이미지 (B,3,224,224) → feat_list[2] = (B, 14, 14, 384)
        #    → permute(0,3,1,2) 통해 (B, 384, 14, 14) 로 변경
        # ─────────────────────────────────────────────────
        self.swin_feats = create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            features_only=True,
            out_indices=(2,),  # SwinTiny stage2 feature (14×14)
        )
        # Stage2에서 뽑히는 채널 수 (예: 384)
        self.swin3_out_channels = self.swin_feats.feature_info.channels()[0]
        # 예: self.swin3_out_channels == 384

        # ─────────────────────────────────────────────────
        # 3) Concat 후 1×1 Conv로 채널을 줄이는 부분
        #    fused_channels = eff3_out_channels + swin3_out_channels
        #    target_channels = 336  (논문 예시대로 사용)
        # ─────────────────────────────────────────────────
        fused_channels = self.eff3_out_channels + self.swin3_out_channels
        target_channels = 336  # Stage3에서 원하는 최종 채널 수

        self.reduce_conv = nn.Conv2d(in_channels=fused_channels, out_channels=target_channels, kernel_size=1, bias=False)
        self.reduce_bn = nn.BatchNorm2d(target_channels)
        self.reduce_relu = nn.ReLU(inplace=True)

        # ─────────────────────────────────────────────────
        # 4) ResCBAM 모듈 (in_channels = target_channels, reduction=16)
        # ─────────────────────────────────────────────────
        self.res_cbam = ResCBAM(in_channels=target_channels, reduction=16)

        # ─────────────────────────────────────────────────
        # 5) Dropout2d (p=0.03)
        # ─────────────────────────────────────────────────
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224) — “원본 이미지”를 그대로 받아옵니다.

        # ─────────────────────────────────────────────────
        # 1) EfficientNet Stage3 feature 추출
        #    → eff3_feats: (B, eff3_out_channels, 14, 14)
        # ─────────────────────────────────────────────────
        eff3_feats = self.efficient_feats(x)[0]  # → (B, C_eff3, 14, 14)

        # ─────────────────────────────────────────────────
        # 2) SwinTiny Stage2 feature 추출
        #    → swin3_feats_tmp: (B, 14, 14, 384)
        #    → permute → (B, 384, 14, 14)
        # ─────────────────────────────────────────────────
        swin3_feats_tmp = self.swin_feats(x)[0]  # (B, 14, 14, C_swin3)
        swin3_feats = swin3_feats_tmp.permute(0, 3, 1, 2)  # → (B, C_swin3, 14, 14)

        # ─────────────────────────────────────────────────
        # 3) Concatenate (채널 방향) → (B, fused_channels, 14, 14)
        # ─────────────────────────────────────────────────
        fused = torch.cat([eff3_feats, swin3_feats], dim=1)
        # → fused_channels = self.eff3_out_channels + self.swin3_out_channels

        # ─────────────────────────────────────────────────
        # 4) 1×1 Conv + BN + ReLU → (B, target_channels=336, 14, 14)
        # ─────────────────────────────────────────────────
        out = self.reduce_conv(fused)  # (B, 336, 14, 14)
        out = self.reduce_bn(out)
        out = self.reduce_relu(out)

        # ─────────────────────────────────────────────────
        # 5) ResCBAM → (B, 336, 14, 14)
        # ─────────────────────────────────────────────────
        out = self.res_cbam(out)

        # ─────────────────────────────────────────────────
        # 6) Dropout2d(p=0.03) → (B, 336, 14, 14)
        # ─────────────────────────────────────────────────
        out = self.dropout(out)

        return out  # (B, 336, 14, 14)
