# File: train_utils.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import json
from eye_patch_dataset import EyePatchDataset, EyePatchDatasetInference

scaler = GradScaler()  # Mixed-precision gradient scaler


def set_seed(seed):
    """
    Set random seed for reproducibility (numpy, torch, random).
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_dataset(csv_path, split, seed):
    """
    Split dataset indices for train/val/test according to split ratio.
    Returns dict with index arrays for each split.
    """
    df = pd.read_csv(csv_path)
    idx = np.arange(len(df))
    np.random.default_rng(seed).shuffle(idx)
    n = len(df)
    n_tr = int(n * split[0])
    n_val = int(n * split[1])
    splits = {
        "train": idx[:n_tr],
        "val": idx[n_tr : n_tr + n_val],
        "test": idx[n_tr + n_val :],
    }
    return splits


def get_loaders(csv_path, split, batch_size, seed, split_json):
    """
    Returns PyTorch DataLoaders for train, val, test splits.
    Also saves split indices to JSON for reproducibility.
    """
    splits = split_dataset(csv_path, split, seed)
    with open(split_json, "w") as f:
        json.dump({k: v.tolist() for k, v in splits.items()}, f, indent=2)
    ds_train = EyePatchDataset(csv_path)
    ds_eval = EyePatchDatasetInference(csv_path)
    mk = lambda ds, ids, shuf: DataLoader(Subset(ds, ids), batch_size, shuf, num_workers=4, prefetch_factor=2, pin_memory=True, persistent_workers=True)
    return (mk(ds_train, splits["train"], True), mk(ds_eval, splits["val"], False), mk(ds_eval, splits["test"], False))


def train_one_epoch(model, loader, optimizer, device, train=True):
    """
    Train or evaluate the model for one epoch. Returns average loss.
    """
    model.train(train)
    total_loss = 0.0
    n = 0
    for imgs, lbl in tqdm(loader, leave=False):
        imgs, lbl = imgs.to(device), lbl.to(device)
        with autocast("cuda"):
            pred = model(imgs)
            loss = (pred - lbl).abs().mean()
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total_loss / n


def save_ckpt(model, optimizer, epoch, best, ckpt_path):
    """
    Save model & optimizer state (checkpoint) to file.
    """
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best": best,
        },
        ckpt_path,
    )


def load_ckpt(model, optimizer, ckpt_path, device, lr_head, lr_backbone):
    """
    Load checkpoint. Resets optimizer learning rates.
    Returns (start_epoch, best_metric).
    """
    start_epoch, best = 0, float("inf")
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        print(f"✓ Found checkpoint → {ckpt_path}")
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", 0)
            best = ckpt.get("best", best)
            optimizer.param_groups[0]["lr"] = lr_head
            optimizer.param_groups[1]["lr"] = lr_backbone
            print(f"✓ LR manually reset: {lr_head} / {lr_backbone}")
        print(f"→ Resume from ep.{start_epoch} | prev-best = {best:.2f}px")
    return start_epoch, best


def load_pretrained_weights(model, pretrained_path, device="cpu", verbose=True):
    """
    Load pretrained weights from file into model.
    Handles both state_dict and checkpoint dict format.
    """
    if pretrained_path and os.path.isfile(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location=device)
        # Accept both plain state_dict or dict with 'model' key
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
        if verbose:
            print(f"✓ Loaded pretrained weights from {pretrained_path}")
        return True
    else:
        if verbose:
            print(f"× No pretrained weights found at {pretrained_path}")
        return False
