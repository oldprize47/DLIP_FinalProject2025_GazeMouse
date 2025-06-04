# 파일명: stage2_gif_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from res_cbam import ResCBAM  # 이미 구현해 둔 res_cbam.py 파일이 동일 디렉터리에 있어야 합니다.


class Stage2_GIFModule(nn.Module):
    def __init__(self, dropout_p: float = 0.06):
        super().__init__()

        # ─────────────────────────────────────────────────
        # 1) EfficientNet-B0 Stage2를 위한 설정
        #    - 원본 이미지 (B,3,224,224) → feat_list[2] = (B, 40, 28, 28) 예상
        # ─────────────────────────────────────────────────
        self.efficient_feats = create_model(
            "efficientnet_b0",
            pretrained=False,
            features_only=True,
            out_indices=(2,),  # EfficientNet stage2 feature (28×28)
        )
        # 채널 수를 읽어옵니다. EfficientNet-B0 기준 대략 40 정도가 됩니다.
        self.eff2_out_channels = self.efficient_feats.feature_info.channels()[0]
        # 예: self.eff2_out_channels == 40

        # ─────────────────────────────────────────────────
        # 2) Swin Tiny Stage1을 위한 설정
        #    - 원본 이미지 (B,3,224,224) → feat_list[1] = (B, 28, 28, 192) 형태 반환
        #    → permute(0,3,1,2) 통해 (B,192,28,28) 로 바꿔야 합니다.
        # ─────────────────────────────────────────────────
        self.swin_feats = create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            features_only=True,
            out_indices=(1,),  # SwinTiny stage1 feature (28×28)
        )
        # SwinTiny 기준 채널 수. .feature_info.channels()[0]를 그대로 읽으면 192 정도가 됩니다.
        # 실제 forward 시 permute 이후 이 값이 적용됩니다.
        self.swin2_out_channels = self.swin_feats.feature_info.channels()[0]
        # 예: self.swin2_out_channels == 192

        # ─────────────────────────────────────────────────
        # 3) Concat 후 1×1 Conv로 채널을 줄이는 부분
        #    fused_channels = eff2_out_channels + swin2_out_channels = 40 + 192 = 232
        #    target_channels = 168  (논문 예시대로)
        # ─────────────────────────────────────────────────
        fused_channels = self.eff2_out_channels + self.swin2_out_channels  # 예: 40 + 192 = 232
        target_channels = 168  # Stage2에서 원하는 최종 채널 수

        self.reduce_conv = nn.Conv2d(in_channels=fused_channels, out_channels=target_channels, kernel_size=1, bias=False)
        self.reduce_bn = nn.BatchNorm2d(target_channels)
        self.reduce_relu = nn.ReLU(inplace=True)

        # ─────────────────────────────────────────────────
        # 4) ResCBAM 모듈 (in_channels = target_channels, reduction=16)
        # ─────────────────────────────────────────────────
        self.res_cbam = ResCBAM(in_channels=target_channels, reduction=16)

        # ─────────────────────────────────────────────────
        # 5) Dropout2d (p=0.06)
        # ─────────────────────────────────────────────────
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224) — “원본 이미지”를 그대로 받아옵니다.

        # ─────────────────────────────────────────────────
        # 1) EfficientNet Stage2 feature 추출
        #    → eff2_feats: (B, 40, 28, 28)  (예상)
        # ─────────────────────────────────────────────────
        eff2_feats = self.efficient_feats(x)[0]  # → (B, eff2_out_channels, 28, 28)

        # ─────────────────────────────────────────────────
        # 2) SwinTiny Stage1 feature 추출
        #    → swin2_feats_tmp: (B, 28, 28, 192)
        #    → permute → (B, 192, 28, 28)
        # ─────────────────────────────────────────────────
        swin2_feats_tmp = self.swin_feats(x)[0]  # (B, 28, 28, 192)
        swin2_feats = swin2_feats_tmp.permute(0, 3, 1, 2)  # → (B, 192, 28, 28)

        # ─────────────────────────────────────────────────
        # 3) Concatenate (채널 방향) → (B, 232, 28, 28)
        # ─────────────────────────────────────────────────
        fused = torch.cat([eff2_feats, swin2_feats], dim=1)  # (B, 40+192=232, 28, 28)

        # ─────────────────────────────────────────────────
        # 4) 1×1 Conv + BN + ReLU → (B, 168, 28, 28)
        # ─────────────────────────────────────────────────
        out = self.reduce_conv(fused)  # (B, 168, 28, 28)
        out = self.reduce_bn(out)
        out = self.reduce_relu(out)

        # ─────────────────────────────────────────────────
        # 5) ResCBAM → (B, 168, 28, 28)
        # ─────────────────────────────────────────────────
        out = self.res_cbam(out)

        # ─────────────────────────────────────────────────
        # 6) Dropout2d(p=0.06) → (B, 168, 28, 28)
        # ─────────────────────────────────────────────────
        out = self.dropout(out)

        return out  # (B, 168, 28, 28)
