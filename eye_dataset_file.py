# eye_dataset_file.py

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

# 전처리 정의 (train 스크립트와 동일하게 설정)
face_tf = transforms.Compose(
    [
        transforms.ToPILImage(),  # 이미지를 PIL로 변환
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

eye_tf = transforms.Compose(
    [
        transforms.ToPILImage(),  # 이미지를 PIL로 변환
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class RTGENDatasetEyes(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def _imread_rgb(self, path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 파일 경로 컬럼 이름 수정
        face_path = row["face_path"]
        left_path = row["left_eye_path"]
        right_path = row["right_eye_path"]

        # 이미지 읽기 + 전처리
        face_img = self._imread_rgb(face_path)
        leye_img = self._imread_rgb(left_path)
        reye_img = self._imread_rgb(right_path)

        face = face_tf(face_img)
        leye = eye_tf(leye_img)
        reye = eye_tf(reye_img)

        # 눈 입력을 채널 방향으로 concatenation
        eyes = torch.cat([leye, reye], dim=0)  # (6,112,112)

        # 라벨 로드 (칼럼 이름과 일치)
        # head_yaw, head_pitch, gaze_yaw, gaze_pitch 순
        label = torch.tensor([row["head_yaw"], row["head_pitch"], row["gaze_yaw"], row["gaze_pitch"]], dtype=torch.float32)

        return face, eyes, label
