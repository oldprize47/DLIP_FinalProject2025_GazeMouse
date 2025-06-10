import json, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from tqdm import tqdm

from mpiigaze_screen_dataset import MPIIGazeScreenDataset
from fginet import FGINet  # 얼굴 하나만 쓰는 버전

CSV_PATH = "mpiigaze/mpiigaze_labels.csv"
BATCH_SIZE = 32
EPOCHS = 50
SPLIT = (0.8, 0.1, 0.1)
SEED = 42
CKPT_BEST = "best_fginet_screen.pth"
LR_NEW = 2e-4
LR_BACKBONE = 5e-5
UNFREEZE_EP = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 데이터 로더 ─────────────────────────────────────────
def make_loaders():
    ds = MPIIGazeScreenDataset(CSV_PATH)
    N = len(ds)
    n_tr, n_val = int(N * SPLIT[0]), int(N * SPLIT[1])
    idx = list(range(N))
    np.random.default_rng(SEED).shuffle(idx)
    splits = {"train": idx[:n_tr], "val": idx[n_tr : n_tr + n_val], "test": idx[n_tr + n_val :]}
    json.dump(splits, open("split_screen.json", "w"))

    mk = lambda ids, shuf: DataLoader(Subset(ds, ids), BATCH_SIZE, shuf, num_workers=4, pin_memory=True, persistent_workers=True)
    return mk(splits["train"], True), mk(splits["val"], False)


# ── Loss (가중 L1) ─────────────────────────────────────
SCREEN_AR = 1280 / 800  # 1280×800 기준
LAMBDA_Y = SCREEN_AR


def weighted_l1(pred, gt):
    dx = torch.abs(pred[:, 0] - gt[:, 0])
    dy = torch.abs(pred[:, 1] - gt[:, 1]) * LAMBDA_Y
    return (dx + dy).mean()


# ── 학습 / 평가 루프 ────────────────────────────────────
def run_epoch(model, loader, train, opt=None):
    model.train(train)
    total, n = 0.0, 0
    for imgs, lbl in tqdm(loader, leave=False):
        imgs, lbl = imgs.to(device), lbl.to(device)
        with torch.set_grad_enabled(train):
            pred = model(imgs)
            loss = weighted_l1(pred, lbl)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
        total += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total / n


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    tr_ld, vl_ld = make_loaders()
    model = FGINet().to(device)

    # 백본 Freeze
    for n, p in model.named_parameters():
        if "efficient_feats" in n or "swin_feats" in n:
            p.requires_grad = False

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_NEW, weight_decay=1e-4)

    best = 1e9
    for ep in range(EPOCHS):
        tqdm.write(f"\n=== Epoch {ep+1}/{EPOCHS} ===")

        # Unfreeze
        if ep == UNFREEZE_EP:
            tqdm.write(" → Backbone unfrozen")
            for n, p in model.named_parameters():
                if "efficient_feats" in n or "swin_feats" in n:
                    p.requires_grad = True
            opt.add_param_group({"params": [p for p in model.parameters() if p.requires_grad and not any(p in g["params"] for g in opt.param_groups)], "lr": LR_BACKBONE})

        tr = run_epoch(model, tr_ld, True, opt)
        vl = run_epoch(model, vl_ld, False)
        tqdm.write(f"train {tr:.4f} | val {vl:.4f}")

        if vl < best:
            best = vl
            torch.save(model.state_dict(), CKPT_BEST)
            tqdm.write(" ✓ best saved")


if __name__ == "__main__":
    main()
