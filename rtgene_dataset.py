# 파일명: rtgene_dataset.py

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class RTGENDataset(Dataset):
    """
    - pairs_csv: img_path, eye_pitch, eye_yaw 세 컬럼이 있는 CSV 파일 경로
    - transform: torchvision.transforms 형태의 전처리
    """
    def __init__(self, pairs_csv: str, transform=None):
        super().__init__()
        self.data = pd.read_csv(pairs_csv)
        # CSV 컬럼: "img_path", "eye_pitch", "eye_yaw"
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row["img_path"]
        eye_pitch = float(row["eye_pitch"])
        eye_yaw   = float(row["eye_yaw"])

        # 이미지 로드
        image = Image.open(img_path).convert("RGB")

        # 전처리(transform) 적용
        if self.transform is not None:
            image = self.transform(image)

        # 레이블 생성
        label = torch.tensor([eye_pitch, eye_yaw], dtype=torch.float32)
        return image, label
