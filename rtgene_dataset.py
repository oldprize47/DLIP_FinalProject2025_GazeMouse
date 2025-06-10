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

    def __init__(self, pairs_csv, transform):
        df = pd.read_csv(pairs_csv)

        self.paths = df["img_path"].tolist()  # list → 빠름
        self.head_pitch = df["head_pitch"].values  # NumPy 배열
        self.head_yaw = df["head_yaw"].values
        self.eye_pitch = df["eye_pitch"].values
        self.eye_yaw = df["eye_yaw"].values
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)

        label = torch.tensor([self.head_pitch[idx], self.head_yaw[idx], self.eye_pitch[idx], self.eye_yaw[idx]], dtype=torch.float32)

        return img, label

    def __len__(self):
        return len(self.paths)
