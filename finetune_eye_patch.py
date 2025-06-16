# 파일명: finetune_eye_patch.py
import os, json, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.nn.functional import l1_loss
import torch.optim as optim
from tqdm import tqdm
from time import perf_counter
from torch.amp import autocast, GradScaler
import pyautogui
import pandas as pd
from eye_patch_dataset import EyePatchDataset, EyePatchDatasetInference
from fginet import FGINet

# ----------- 설정 -----------
CSV_PATH = "recalib_data_SH_flipx.csv"  # 파인튜닝용 데이터셋 경로
PRETRAINED_CKPT = "Last_fine_SH.pth"  # 프리트레인(대규모 데이터) 모델
CKPT_BEST = "Last_fine_SH.pth"  # 저장할 파인튜닝 체크포인트
BATCH_SIZE = 16
EPOCHS = 150
SPLIT = (0.9, 0.05, 0.05)  # (train/val/test)
SEED = 47
LR_HEAD = 1e-5
LR_BACKBONE = 3e-6
PATIENCE = 20
W, H = pyautogui.size()
LAMBDA_Y = W / H
# ----------------------------

scaler = GradScaler()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")


def weighted_mse_loss(pred, gt, lam_x=1.0, lam_y=LAMBDA_Y):
    dx = ((pred[:, 0] - gt[:, 0]) ** 2) * lam_x
    dy = ((pred[:, 1] - gt[:, 1]) ** 2) * lam_y
    return (dx + dy).mean()


def mae_loss(pred, gt):
    return (pred - gt).abs().mean()


def run_epoch_mse(model, loader):
    model.eval()
    tot = 0.0
    n = 0
    with torch.no_grad():
        for imgs, lbl in tqdm(loader, leave=False):
            imgs = imgs.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            pred = model(imgs)
            loss = weighted_mse_loss(pred, lbl)
            tot += loss.item() * imgs.size(0)
            n += imgs.size(0)
    return tot / n


def make_loaders(csv_path):
    df = pd.read_csv(csv_path)
    N = len(df)
    n_tr = int(N * SPLIT[0])
    n_val = int(N * SPLIT[1])
    idx = list(range(N))
    np.random.default_rng(SEED).shuffle(idx)
    splits = {"train": idx[:n_tr], "val": idx[n_tr : n_tr + n_val], "test": idx[n_tr + n_val :]}
    json.dump(splits, open("split_eye_patch_finetune.json", "w"))
    ds_train = EyePatchDataset(csv_path)
    ds_eval = EyePatchDatasetInference(csv_path)
    mk = lambda ds, ids, shuf: DataLoader(Subset(ds, ids), BATCH_SIZE, shuf, num_workers=4, prefetch_factor=2, pin_memory=True, persistent_workers=True)
    return mk(ds_train, splits["train"], True), mk(ds_eval, splits["val"], False), mk(ds_eval, splits["test"], False)


def run_epoch(model, loader, train, opt=None):
    model.train(train)
    tot = 0.0
    n = 0
    for imgs, lbl in tqdm(loader, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        lbl = lbl.to(device, non_blocking=True)
        with autocast("cuda"):
            pred = model(imgs)
            loss = mae_loss(pred, lbl)
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss = loss.detach()
        tot += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return tot / n


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 데이터로더
    tr_ld, vl_ld, te_ld = make_loaders(CSV_PATH)

    # ----- 모델 생성 & 프리트레인 가중치만 불러오기 -----
    model = FGINet().to(device)
    ckpt = torch.load(PRETRAINED_CKPT, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    print("✓ Pretrained weights loaded")

    # ---- 옵티마이저/스케줄러 새로 (이전 거 이어받지 X) ----
    head_params, backbone_params = [], []
    for n, p in model.named_parameters():
        if "stage" in n:
            backbone_params.append(p)
        else:
            head_params.append(p)
    opt = optim.AdamW([{"params": head_params, "lr": LR_HEAD, "initial_lr": LR_HEAD}, {"params": backbone_params, "lr": LR_BACKBONE, "initial_lr": LR_BACKBONE}], weight_decay=1e-3, fused=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

    # ------ 학습 루프 ------
    best = float("inf")
    no_imp = 0
    for ep in range(EPOCHS):
        tqdm.write(f"\n=== Epoch {ep+1}/{EPOCHS} ===")
        tqdm.write(f" current LR {opt.param_groups[0]['lr']:.2e}")
        tr_loss = run_epoch(model, tr_ld, True, opt)
        vl_loss = run_epoch(model, vl_ld, False)
        scheduler.step()
        tqdm.write(f" train {tr_loss:.4f} | val {vl_loss:.4f}")

        if vl_loss < best:
            best = vl_loss
            no_imp = 0
            torch.save(
                {
                    "epoch": ep + 1,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "best": best,
                },
                CKPT_BEST,
            )
            tqdm.write("  ✓ best model saved")
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                tqdm.write(f"Early-stopped (no val improve {PATIENCE} ep)")
                break
    tqdm.write("Training finished. Best val loss: %.4f" % best)

    # ---- 최종 Test ----
    model.load_state_dict(torch.load(CKPT_BEST, map_location=device)["model"])
    model.eval()
    test_loss_mae = run_epoch(model, te_ld, False)
    test_loss_mse = run_epoch_mse(model, te_ld)
    print(f"\n=== Test MSE (px): {test_loss_mse:.2f}")
    print(f"=== Test MAE (px): {test_loss_mae:.2f}")


if __name__ == "__main__":
    main()
