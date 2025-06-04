# 파일명: train_clean_resume.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from rtgene_dataset import RTGENDataset
from fginet import FGINet

from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR, CyclicLR
import random, numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device, use_clr=False, clr_scheduler=None):
    model.train()
    running_loss = 0.0

    # ─── Warm-up이나 CLR에 상관없이, 오직 배치 진행바만 돌립니다. ───
    loop = tqdm(loader, desc="  [Train]", leave=False)
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if use_clr:
            # CLR 구간일 때만 배치마다 LR 스케줄러 호출
            clr_scheduler.step()

        running_loss += loss.item() * images.size(0)
        loop.set_postfix(batch_loss=loss.item())

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    loop = tqdm(loader, desc="  [Val]  ", leave=False)
    with torch.no_grad():
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            loop.set_postfix(val_loss=loss.item())

    return running_loss / len(loader.dataset)


if __name__ == "__main__":
    # ────────────────── 하이퍼파라미터 ──────────────────
    num_epochs = 50  # 워밍업 + CLR 일부만 확인
    batch_size = 16  # GPU 메모리에 맞춰 적절히 변경
    base_lr = 5e-4  # 논문: 워밍업 후 최대 lr = 0.0005
    val_split = 0.1  # 10%를 검증용으로

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ────────────────── 데이터 전처리 ──────────────────
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    pairs_csv = "RT_GENE/rtgene_original_pairs.csv"
    full_dataset = RTGENDataset(pairs_csv=pairs_csv, transform=transform)

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    # 고정 시드 generator
    split_seed = 42
    generator = torch.Generator().manual_seed(split_seed)

    # ★ generator 인자를 꼭 넘겨야 매번 같은 split이 됩니다
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)  # ← 이 부분 추가!

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)  # 학습용은 epoch마다 셔플 OK
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)  # 검증셋은 셔플 X

    # ────────────────── 모델·손실·옵티마이저 정의 ──────────────────
    model = FGINet().to(device)
    criterion = nn.L1Loss()

    # ─── optimizer 초기 lr = base_lr(0.0005) ───
    optimizer = optim.Adam(model.parameters(), lr=base_lr, betas=(0.88, 0.999), weight_decay=0.0)

    # ─── 1) Warm-up 스케줄러 (Epoch 1~5) ───
    def warmup_lambda(epoch):
        return float(epoch + 1) / 5.0 if epoch < 5 else 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # ─── 2) CLR 스케줄러 (Epoch ≥6) ───
    step_size_up = len(train_loader) * 5  # 배치 수 × 5 에폭
    clr_scheduler = CyclicLR(optimizer, base_lr=1e-4, max_lr=base_lr, step_size_up=step_size_up, step_size_down=step_size_up, mode="triangular", cycle_momentum=False)  # CLR 최저 lr = 0.0001  # CLR 최고 lr = 0.0005

    # ────────────────── 체크포인트 로드 ──────────────────
    checkpoint_path = "best_fginet.pth"
    start_epoch = 0
    best_val_loss = float("inf")

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        # 모델 가중치 복원
        model.load_state_dict(ckpt["model_state_dict"])
        # 옵티마이저 상태 복원
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # 이전 최저 val_loss, 다음 에폭 번호 복원
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        start_epoch = ckpt.get("epoch", 0)
        tqdm.write(f"Loaded checkpoint (epoch {start_epoch}, best_val_loss {best_val_loss:.6f})")
    else:
        tqdm.write("No checkpoint found, starting from scratch.")

    # ────────────────── 학습 루프 ──────────────────
    for epoch in range(start_epoch, num_epochs):
        tqdm.write(f"\n=== Epoch {epoch+1}/{num_epochs} ===")

        if epoch < 5:
            # ─── ① Warm-up 구간 (Epoch 1~5) ───
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, use_clr=False, clr_scheduler=None)  # CLR 사용 안 함  # CLR 스케줄러 전달 X
            warmup_scheduler.step()  # 에폭마다 lr 업데이트

            # (원한다면 Epoch 시작 전 lr을 보고싶다면 다음 코드 추가)
            # current_lr = optimizer.param_groups[0]["lr"]
            # tqdm.write(f"(Warm-up)   After Epoch {epoch+1}, lr = {current_lr:.6f}")

        else:
            # ─── ② CLR 구간 (Epoch 6~7) ───
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, use_clr=True, clr_scheduler=clr_scheduler)  # CLR 사용
            # Epoch 종료 시 lr 출력 (선택 사항)
            # current_lr = optimizer.param_groups[0]["lr"]
            # tqdm.write(f"(CLR)       After Epoch {epoch+1}, lr = {current_lr:.6f}")

        # ─── 검증 ───
        val_loss = validate(model, val_loader, criterion, device)
        tqdm.write(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

        # ─── 체크포인트 저장 조건 ───
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 모델·옵티마이저 상태 + epoch + best_val_loss 기록
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "best_val_loss": best_val_loss}, checkpoint_path)
            tqdm.write(f"  → Saved checkpoint (epoch {epoch+1}, val_loss: {val_loss:.4f})")

    tqdm.write("Training complete.")
