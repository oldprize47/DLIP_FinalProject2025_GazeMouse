# 파일: train_eye_patch.py
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

# ── 설정 ────────────────────────────────────────────────
CSV_PATH = "mpiigaze/mpiigaze_labels_center_px.csv"
BATCH_SIZE = 16
EPOCHS = 50
SPLIT = (0.8, 0.1, 0.1)
SEED = 47
CKPT_BEST = "Last_MPIIGAZE.pth"

LR_HEAD = 1e-3
LR_BACKBONE = 3e-4
WARM_EPOCHS = 2
PATIENCE = 15  # early stop

W, H = pyautogui.size()
LAMBDA_Y = W / H
# --------------------------------------------------------

scaler = GradScaler()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")


def benchmark(loader, model, max_batches=100):
    model.eval()
    t_data = t_infer = 0.0
    with torch.no_grad():
        for i, (imgs, lbl) in enumerate(loader):
            if i >= max_batches:
                break
            t0 = perf_counter()
            imgs = imgs.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = perf_counter()
            _ = model(imgs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = perf_counter()
            t_data += t1 - t0
            t_infer += t2 - t1
    n = i + 1
    print(f"[per batch] Data {t_data/n:.4f}s | Infer {t_infer/n:.4f}s")


def weighted_mse_loss(pred, gt, lam_x=1.0, lam_y=LAMBDA_Y):
    dx = ((pred[:, 0] - gt[:, 0]) ** 2) * lam_x
    dy = ((pred[:, 1] - gt[:, 1]) ** 2) * lam_y
    return (dx + dy).mean()


def mae_loss(pred, gt):
    return (pred - gt).abs().mean()


def run_epoch_mae(model, loader):
    model.eval()
    tot = 0.0
    n = 0
    with torch.no_grad():
        for imgs, lbl in tqdm(loader, leave=False):
            imgs = imgs.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            pred = model(imgs)
            loss = mae_loss(pred, lbl)
            tot += loss.item() * imgs.size(0)
            n += imgs.size(0)
    return tot / n


def l1_center(pred, gt):
    return l1_loss(pred, gt)


def make_loaders(csv_path):
    # 전체 데이터 인덱스 분할
    df = pd.read_csv(csv_path)
    N = len(df)
    n_tr = int(N * SPLIT[0])
    n_val = int(N * SPLIT[1])
    idx = list(range(N))
    np.random.default_rng(SEED).shuffle(idx)
    splits = {"train": idx[:n_tr], "val": idx[n_tr : n_tr + n_val], "test": idx[n_tr + n_val :]}
    json.dump(splits, open("split_eye_patch.json", "w"))

    # 각각 다른 Dataset 사용!
    ds_train = EyePatchDataset(csv_path)  # 증강 포함 (train)
    ds_eval = EyePatchDatasetInference(csv_path)  # 증강 없음 (val, test)
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
            loss = weighted_mse_loss(pred, lbl)
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

    # 1) 데이터로더
    tr_ld, vl_ld, te_ld = make_loaders(CSV_PATH)

    # 2) 모델·옵티마이저
    model = FGINet().to(device)
    # head / backbone 파라미터 분리
    head_params, backbone_params = [], []
    for n, p in model.named_parameters():
        if "stage" in n:
            backbone_params.append(p)
        else:
            head_params.append(p)
    opt = optim.AdamW([{"params": head_params, "lr": LR_HEAD, "initial_lr": LR_HEAD}, {"params": backbone_params, "lr": LR_BACKBONE, "initial_lr": LR_BACKBONE}], weight_decay=1e-3, fused=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS - WARM_EPOCHS, eta_min=1e-5)

    # 3) 체크포인트 로드
    start_epoch = 0
    best = float("inf")

    if os.path.isfile(CKPT_BEST):
        ckpt = torch.load(CKPT_BEST, map_location=device)
        print(f"✓ Found checkpoint  →  {CKPT_BEST}")
        # 1) 모델 가중치
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)

        # 2) 옵티마이저 / 스케줄러 (있을 때만)
        if isinstance(ckpt, dict) and "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", 0)  # 저장했던 epoch+1
            best = ckpt.get("best", best)  # 저장했던 val best
            # 스케줄러 epoch 보정
            scheduler.last_epoch = start_epoch - 1
            # ★★★ 여기서 learning rate를 원래 값으로 다시 "수동" 지정해준다!
            opt.param_groups[0]["lr"] = LR_HEAD  # head
            opt.param_groups[1]["lr"] = LR_BACKBONE  # backbone
            print(f"✓ LR manually reset: {LR_HEAD} / {LR_BACKBONE}")

        print(f"→ Resume from ep.{start_epoch}  |  prev-best = {best:.2f}px")

    # 4) 학습 루프

    no_imp = 0
    for ep in range(start_epoch, EPOCHS):
        tqdm.write(f"\n=== Epoch {ep+1}/{EPOCHS} ===")
        tqdm.write(f" current LR {opt.param_groups[0]['lr']:.2e}")

        tr_loss = run_epoch(model, tr_ld, True, opt)
        vl_loss = run_epoch(model, vl_ld, False)

        # warm-up or scheduler
        if ep < WARM_EPOCHS:
            scale = (ep + 1) / WARM_EPOCHS
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * scale
        else:
            scheduler.step()

        tqdm.write(f" train {tr_loss:.4f} | val {vl_loss:.4f}")

        # best 저장 (model + optimizer + epoch)
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

    # 5) 최종 Test
    model.load_state_dict(torch.load(CKPT_BEST, map_location=device)["model"])
    model.eval()
    test_loss_MSE = run_epoch(model, te_ld, False)
    test_loss_mae = run_epoch_mae(model, te_ld)
    print(f"\n=== Test MSE (px): {test_loss_MSE:.2f}")
    print(f"=== Test MAE (px): {test_loss_mae:.2f}")


if __name__ == "__main__":
    main()
