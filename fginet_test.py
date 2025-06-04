# 파일명: fginet_test.py

import torch
from fginet import FGINet

if __name__ == "__main__":
    # 더미 입력: 배치 2, 3채널, 224×224
    dummy = torch.randn(2, 3, 224, 224)

    # FGI-Net 모델 생성
    model = FGINet()

    # Forward
    out = model(dummy)

    # 출력 shape 확인
    print("FGI-Net output shape:", out.shape)
    # 예상: torch.Size([2, 2])