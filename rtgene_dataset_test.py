# 파일명: rtgene_dataset_test.py

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from rtgene_dataset import RTGENDataset

if __name__ == "__main__":
    # 1) 이미지 전처리 설정 (Resize → ToTensor)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # 필요하다면 Normalize를 추가하세요
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225]),
        ]
    )

    # 2) Dataset 생성 (CSV 경로를 본인 경로로 수정)
    pairs_csv = "RT_GENE/rtgene_original_pairs.csv"
    dataset = RTGENDataset(pairs_csv=pairs_csv, transform=transform)

    # 3) DataLoader 생성
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    # 4) 한 배치만 추출하여 shape 확인
    images, labels = next(iter(loader))
    print("Images batch shape:", images.shape)  # e.g. (8, 3, 224, 224)
    print("Labels batch shape:", labels.shape)  # e.g. (8, 2)
    print("첫 번째 레이블 예시 (eye_pitch, eye_yaw):", labels[0])
