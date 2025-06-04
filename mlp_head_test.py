# 파일명: mlp_head_test.py

import torch
from mlp_head import MLPHead

if __name__ == "__main__":
    # 1) 더미 텐서 (Stage3 출력 형태와 동일하게): 배치 2, 채널 336, 크기 14×14
    dummy_stage3 = torch.randn(2, 336, 14, 14)

    # 2) MLP Head 생성 (in_features=336)
    model = MLPHead(in_features=336, hidden_dim=32, out_dim=2, dropout_p=0.1)

    # 3) Forward
    out = model(dummy_stage3)

    # 4) 출력 shape 확인
    print("MLPHead output shape:", out.shape)
    # 예상: torch.Size([2, 2])
