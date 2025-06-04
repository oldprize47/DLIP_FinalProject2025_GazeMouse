# stage1_test_module.py

import torch
from stage1_gif_module import Stage1_GIFModule

if __name__ == "__main__":
    dummy = torch.randn(2, 3, 224, 224)
    model = Stage1_GIFModule(dropout_p=0.1)
    out = model(dummy)
    print("Stage1 output shape:", out.shape)
    # 예상: torch.Size([2, 96, 56, 56])
