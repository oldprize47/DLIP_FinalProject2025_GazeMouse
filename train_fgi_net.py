# 파일: train_eyes.py
import os, time, random, json, numpy as np, torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from eye_dataset_file import RTGENDatasetEyes  # 얼굴+눈 데이터
from fginet_eyes import FGINetEyes  # 새 모델

# ── 설정 ────────────────────────────────────────────────
CSV_PATH = "RT_GENE/rtgene_pairs_face_eye.csv"
BATCH_SIZE = 32
EPOCHS = 40
SPLIT = (0.8, 0.1, 0.1)  # train:val:test
SEED = 42
CKPT_BEST = "best_fgineteyes.pth"
SPLIT_JSON = "split_indices.json"

LR_NEW = 1e-4  # 눈 브랜치
LR_BACKBONE = 3e-5  # 백본(6 epoch 이후)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------------------------------------------


def angular_loss(pred, gt, eps=1e-6):
    gx, gy = torch.sin(pred[:, 0]), torch.sin(pred[:, 1])
    gz = torch.cos(pred[:, 0]) * torch.cos(pred[:, 1])
    vx, vy = torch.sin(gt[:, 0]), torch.sin(gt[:, 1])
    vz = torch.cos(gt[:, 0]) * torch.cos(gt[:, 1])
    return torch.acos((gx * vx + gy * vy + gz * vz).clamp(-1 + eps, 1 - eps)).mean()


def loss_fn(pred, gt):
    head = nn.functional.l1_loss(pred[:, :2], gt[:, :2])
    gaze = angular_loss(pred[:, 2:], gt[:, 2:])
    return head + 2.0 * gaze


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def make_loaders():
    full_ds = RTGENDatasetEyes(CSV_PATH)
    N = len(full_ds)
    n_train = int(N * SPLIT[0])
    n_val = int(N * SPLIT[1])
    n_test = N - n_train - n_val

    idx_all = list(range(N))
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx_all)

    train_idx = idx_all[:n_train]
    val_idx = idx_all[n_train : n_train + n_val]
    test_idx = idx_all[n_train + n_val :]

    # split 정보 저장 → 나중 시각화 때 그대로 사용
    with open(SPLIT_JSON, "w") as f:
        json.dump({"train": train_idx, "val": val_idx, "test": test_idx}, f)

    subset = torch.utils.data.Subset
    train_ld = DataLoader(subset(full_ds, train_idx), BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_ld = DataLoader(subset(full_ds, val_idx), BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_ld = DataLoader(subset(full_ds, test_idx), BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return train_ld, val_ld, test_ld


# ── main ───────────────────────────────────────────────
def main():
    set_seed(SEED)
    train_ld, val_ld, test_ld = make_loaders()

    model = FGINetEyes().to(device)
    for p in model.backbone.parameters():
        p.requires_grad = False  # 눈 브랜치만 학습
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_NEW, weight_decay=1e-4)

    def run(loader, train=True):
        model.train(train)
        tot, acc = 0, 0.0
        for face, eyes, y in tqdm(loader, leave=False):
            face, eyes, y = face.to(device), eyes.to(device), y.to(device)
            with torch.set_grad_enabled(train):
                pred = model(face, eyes)
                loss = loss_fn(pred, y)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            acc += loss.item() * face.size(0)
            tot += face.size(0)
        return acc / tot

    best = 1e9
    for ep in range(EPOCHS):
        tqdm.write(f"\n=== Epoch {ep+1}/{EPOCHS} ===")
        if ep == 6:  # 백본 언프리즈
            for p in model.backbone.parameters():
                p.requires_grad = True
            opt.add_param_group({"params": model.backbone.parameters(), "lr": LR_BACKBONE})
            tqdm.write("→ Backbone unfrozen")

        tr = run(train_ld, True)
        vl = run(val_ld, False)
        tqdm.write(f"train {tr:.4f} | val {vl:.4f}")

        if vl < best:
            best = vl
            torch.save(model.state_dict(), CKPT_BEST)
            tqdm.write("  ✓ best model saved")

    tqdm.write("Training finished. Best val loss: %.4f" % best)


if __name__ == "__main__":
    main()
