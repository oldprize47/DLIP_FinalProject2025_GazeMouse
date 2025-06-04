from timm import create_model
import torch

# Swin Tiny을 features_only=True로 불러오기 (pretrained=False 상태)
swin = create_model("swin_tiny_patch4_window7_224", pretrained=False, features_only=True, out_indices=(0, 1, 2, 3))  # stage0~stage3 모두 뽑아서 확인

# 더미 입력
x = torch.randn(2, 3, 224, 224)
features = swin(x)

for i, feat in enumerate(features):
    feat = feat.permute(0, 3, 1, 2)
    print(f"Swin stage{i} feature shape:", feat.shape)
