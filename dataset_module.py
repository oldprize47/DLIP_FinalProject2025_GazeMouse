import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import cv2


class GazeScreenDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        # 1) CSV 로드
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 2) 이미지 읽기 & 전처리
        img = cv2.imread(row.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        if self.transform:
            img = self.transform(img)
        else:
            # 기본: [H,W,C] → [C,H,W], 0–255 → 0–1
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # 3) 레이블: 정규화된 스크린 좌표
        label = torch.tensor([row.norm_x, row.norm_y], dtype=torch.float32)
        return img, label


if __name__ == "__main__":
    # (Windows용) freeze_support() 호출
    from multiprocessing import freeze_support

    freeze_support()

    csv_path = "C:/Users/Sangheon/source/repos/DLIP/DLIP_FinalProject2025_GazeMouse/mpiigaze/mpiigaze_labels.csv"
    dataset = GazeScreenDataset(csv_path)
    # 저는 우선 워커 없이(num_workers=0) 테스트했습니다
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    imgs, labels = next(iter(loader))
    print(imgs.shape, labels.shape)
