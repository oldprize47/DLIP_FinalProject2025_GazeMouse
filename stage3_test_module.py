# 파일명: stage3_test_module.py

import torch
from stage3_gif_module import Stage3_GIFModule

if __name__ == "__main__":
    # 더미 입력 이미지: 배치 2, 3채널, 224×224
    dummy = torch.randn(2, 3, 224, 224)

    # Stage3 모듈 생성
    model = Stage3_GIFModule(dropout_p=0.03)

    # Forward
    out = model(dummy)

    # 출력 shape 확인
    print("Stage3 output shape:", out.shape)
    # 예상: torch.Size([2, 336, 14, 14])
