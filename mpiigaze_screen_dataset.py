# 파일: mpiigaze_screen_dataset.py
import pandas as pd, torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode


class MPIIGazeScreenDataset(torch.utils.data.Dataset):
    """
    CSV 열:
    subject, day, image_path, screen_x, screen_y, norm_x, norm_y
    ───────────── image_path ──────────────  ─── label ───
    """

    def __init__(self, csv_path, img_size=224):
        df = pd.read_csv(csv_path, sep="\t" if csv_path.endswith(".tsv") else ",")
        self.paths = df["image_path"].tolist()
        self.labels = df[["norm_x", "norm_y"]].values.astype("float32")

        # ② Pad→Resize 로 종횡비 보존 + 가벼운 RandomResizedCrop
        self.tf = transforms.Compose([transforms.Lambda(self._pad_to_square), transforms.RandomResizedCrop(img_size, scale=(0.95, 1.0), ratio=(1.0, 1.5), interpolation=InterpolationMode.BILINEAR), transforms.ColorJitter(brightness=0.2, contrast=0.2), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # ③ helper: 짧은 변 기준 흰색(255) 패딩 → 정사각형
    @staticmethod
    def _pad_to_square(img):
        w, h = img.size
        if w == h:
            return img
        L = R = (h - w) // 2 if h > w else 0
        T = B = (w - h) // 2 if w > h else 0
        pad = (L, T, R + (h - w) % 2, B + (w - h) % 2)
        return F.pad(img, pad, fill=255, padding_mode="constant")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.tf(img)
        label = torch.tensor(self.labels[idx])  # (2,)
        return img, label
