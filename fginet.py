# 파일명: fginet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model


# ─────────────────────────────────────────────
# 1. Attention 모듈 (CBAM)
# ─────────────────────────────────────────────


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_channels, in_channels // reduction, bias=True), nn.ReLU(inplace=True), nn.Linear(in_channels // reduction, in_channels, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(B, C)
        max_pool = F.adaptive_max_pool2d(x, 1).view(B, C)
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        attn = self.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(concat))
        return x * attn


class ResCBAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.pw_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(7)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn(self.dw_conv(x)))
        out = self.relu(self.bn(self.pw_conv(out)))
        out = self.channel_att(out)
        out = self.spatial_att(out)
        out = self.relu(out + residual)
        return out


# ─────────────────────────────────────────────
# 2. Stage별 GIF 모듈 (EfficientNet + Swin + CBAM)
# ─────────────────────────────────────────────


class Stage1_GIFModule(nn.Module):
    def __init__(self, dropout_p: float = 0.1):
        super().__init__()
        # EfficientNet-B0 stage1 (56x56)
        self.efficient_feats = create_model("efficientnet_b0", pretrained=True, features_only=True, out_indices=(1,))
        self.eff_out_channels = self.efficient_feats.feature_info.channels()[0]
        # SwinTiny stage0 (56x56)
        self.swin_feats = create_model("swin_tiny_patch4_window7_224", pretrained=True, features_only=True, out_indices=(0,))
        self.swin_out_channels = self.swin_feats.feature_info.channels()[0]
        fused_channels = self.eff_out_channels + self.swin_out_channels
        target_channels = 96
        self.reduce_conv = nn.Conv2d(fused_channels, target_channels, 1, bias=False)
        self.reduce_bn = nn.BatchNorm2d(target_channels)
        self.reduce_relu = nn.ReLU(inplace=True)
        self.res_cbam = ResCBAM(target_channels, reduction=16)
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        eff_feats = self.efficient_feats(x)[0]
        swin_feats = self.swin_feats(x)[0].permute(0, 3, 1, 2)
        fused = torch.cat([eff_feats, swin_feats], dim=1)
        out = self.reduce_relu(self.reduce_bn(self.reduce_conv(fused)))
        out = self.res_cbam(out)
        out = self.dropout(out)
        return out  # (B, 96, 56, 56)


class Stage2_GIFModule(nn.Module):
    def __init__(self, dropout_p: float = 0.06):
        super().__init__()
        self.efficient_feats = create_model("efficientnet_b0", pretrained=True, features_only=True, out_indices=(2,))
        self.eff2_out_channels = self.efficient_feats.feature_info.channels()[0]
        self.swin_feats = create_model("swin_tiny_patch4_window7_224", pretrained=True, features_only=True, out_indices=(1,))
        self.swin2_out_channels = self.swin_feats.feature_info.channels()[0]
        fused_channels = self.eff2_out_channels + self.swin2_out_channels
        target_channels = 168
        self.reduce_conv = nn.Conv2d(fused_channels, target_channels, 1, bias=False)
        self.reduce_bn = nn.BatchNorm2d(target_channels)
        self.reduce_relu = nn.ReLU(inplace=True)
        self.res_cbam = ResCBAM(target_channels, reduction=16)
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        eff2_feats = self.efficient_feats(x)[0]
        swin2_feats = self.swin_feats(x)[0].permute(0, 3, 1, 2)
        fused = torch.cat([eff2_feats, swin2_feats], dim=1)
        out = self.reduce_relu(self.reduce_bn(self.reduce_conv(fused)))
        out = self.res_cbam(out)
        out = self.dropout(out)
        return out  # (B, 168, 28, 28)


class Stage3_GIFModule(nn.Module):
    def __init__(self, dropout_p: float = 0.03):
        super().__init__()
        self.efficient_feats = create_model("efficientnet_b0", pretrained=True, features_only=True, out_indices=(3,))
        self.eff3_out_channels = self.efficient_feats.feature_info.channels()[0]
        self.swin_feats = create_model("swin_tiny_patch4_window7_224", pretrained=True, features_only=True, out_indices=(2,))
        self.swin3_out_channels = self.swin_feats.feature_info.channels()[0]
        fused_channels = self.eff3_out_channels + self.swin3_out_channels
        target_channels = 336
        self.reduce_conv = nn.Conv2d(fused_channels, target_channels, 1, bias=False)
        self.reduce_bn = nn.BatchNorm2d(target_channels)
        self.reduce_relu = nn.ReLU(inplace=True)
        self.res_cbam = ResCBAM(target_channels, reduction=16)
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        eff3_feats = self.efficient_feats(x)[0]
        swin3_feats = self.swin_feats(x)[0].permute(0, 3, 1, 2)
        fused = torch.cat([eff3_feats, swin3_feats], dim=1)
        out = self.reduce_relu(self.reduce_bn(self.reduce_conv(fused)))
        out = self.res_cbam(out)
        out = self.dropout(out)
        return out  # (B, 336, 14, 14)


# ─────────────────────────────────────────────
# 3. MLP Head (좌표 예측)
# ─────────────────────────────────────────────


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
        return self.fc(x)  # (B,2)


# ─────────────────────────────────────────────
# 4. FGINet 본체
# ─────────────────────────────────────────────


class FGINet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = Stage1_GIFModule(dropout_p=0.09)
        self.stage2 = Stage2_GIFModule(dropout_p=0.06)
        self.stage3 = Stage3_GIFModule(dropout_p=0.03)
        self.mlp_head = MLPHead(in_features=600, hidden_dim=32, out_dim=2, dropout_p=0.1)

    def forward(self, x):
        x1 = self.stage1(x)  # (B, 96, 56, 56)
        x2 = self.stage2(x)  # (B, 168, 28, 28)
        x3 = self.stage3(x)  # (B, 336, 14, 14)
        g1 = F.adaptive_avg_pool2d(x1, 1).flatten(1)
        g2 = F.adaptive_avg_pool2d(x2, 1).flatten(1)
        g3 = F.adaptive_avg_pool2d(x3, 1).flatten(1)
        feat = torch.cat([g1, g2, g3], dim=1)  # (B, 600)
        out = self.mlp_head(feat)  # (B, 2)
        return out


# ─────────────────────────────────────────────
# 5. (선택) 단독 테스트용
# ─────────────────────────────────────────────

if __name__ == "__main__":
    model = FGINet()
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("FGINet output shape:", out.shape)  # → torch.Size([2, 2])
