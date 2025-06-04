# 파일명: stage2_test_module.py

import torch
from stage2_gif_module import Stage2_GIFModule

if __name__ == "__main__":
    # 더미 입력 이미지: 배치 2, 3채널, 224×224
    dummy = torch.randn(2, 3, 224, 224)

    # Stage2 모듈 생성
    model = Stage2_GIFModule(dropout_p=0.06)

    # Forward
    out = model(dummy)

    # 출력 shape 확인
    print("Stage2 output shape:", out.shape)
    # 예상: torch.Size([2, 168, 28, 28])
