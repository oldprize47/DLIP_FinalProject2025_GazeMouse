# File: eye_patch_dataset.py

import pandas as pd
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class EyePatchDataset(torch.utils.data.Dataset):
    """
    Training dataset.
    Uses only brightness/color augmentation (no rotation/flip).
    Applies smart_crop to remove black borders.
    """

    def __init__(self, csv_path: str, img_size: int = 224):
        # Support both CSV and TSV
        df = pd.read_csv(csv_path, sep="\t" if csv_path.endswith(".tsv") else ",")
        self.paths = df["image_path"].tolist()
        self.labels = df[["dx", "dy"]].values.astype("float32")
        self.img_size = img_size

        # Transform: crop, color jitter, pad to square, tensor, normalize
        self.tf = transforms.Compose(
            [
                transforms.Lambda(self._smart_crop),  # Remove black borders
                transforms.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.07, hue=0.02),
                transforms.Lambda(self._pad_to_square),  # Pad to square
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _smart_crop(img: Image.Image) -> Image.Image:
        """
        Crop out black borders (pixels where sum < threshold).
        """
        arr = np.array(img)
        mask = arr.sum(-1) > 10
        if mask.any():
            ys, xs = np.where(mask)
            x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
            return img.crop((x1, y1, x2 + 1, y2 + 1))
        return img

    def _pad_to_square(self, img: Image.Image) -> Image.Image:
        """
        Resize and pad image to square with black background.
        """
        img_size = self.img_size
        w, h = img.size
        scale = img_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        new_img = Image.new("RGB", (img_size, img_size), (0, 0, 0))
        x_off = (img_size - new_w) // 2
        y_off = (img_size - new_h) // 2
        new_img.paste(img, (x_off, y_off))
        return new_img

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.tf(img)
        label = torch.tensor(self.labels[idx])
        return img, label


class EyePatchDatasetInference(torch.utils.data.Dataset):
    """
    For validation, test, and real-time inference.
    Only uses smart_crop and padding (no augmentation).
    """

    def __init__(self, csv_path, img_size=224):
        df = pd.read_csv(csv_path)
        self.paths = df["image_path"].tolist()
        self.labels = df[["dx", "dy"]].values.astype("float32")
        self.img_size = img_size
        self.tf = transforms.Compose(
            [
                transforms.Lambda(self._smart_crop),
                transforms.Lambda(self._pad_to_square),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _smart_crop(img: Image.Image) -> Image.Image:
        arr = np.array(img)
        mask = arr.sum(-1) > 10
        if mask.any():
            ys, xs = np.where(mask)
            x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
            return img.crop((x1, y1, x2 + 1, y2 + 1))
        return img

    def _pad_to_square(self, img: Image.Image) -> Image.Image:
        img_size = self.img_size
        w, h = img.size
        scale = img_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        new_img = Image.new("RGB", (img_size, img_size), (0, 0, 0))
        x_off = (img_size - new_w) // 2
        y_off = (img_size - new_h) // 2
        new_img.paste(img, (x_off, y_off))
        return new_img

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.tf(img)
        label = torch.tensor(self.labels[idx])
        return img, label


def get_infer_transform(img_size=224):
    """
    Returns image transform pipeline for inference.
    """

    def _smart_crop(img: Image.Image) -> Image.Image:
        arr = np.array(img)
        mask = arr.sum(-1) > 10
        if mask.any():
            ys, xs = np.where(mask)
            x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
            return img.crop((x1, y1, x2 + 1, y2 + 1))
        return img

    def _pad_to_square(img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = img_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        new_img = Image.new("RGB", (img_size, img_size), (0, 0, 0))
        x_off = (img_size - new_w) // 2
        y_off = (img_size - new_h) // 2
        new_img.paste(img, (x_off, y_off))
        return new_img

    return transforms.Compose(
        [
            transforms.Lambda(_smart_crop),
            transforms.Lambda(_pad_to_square),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
