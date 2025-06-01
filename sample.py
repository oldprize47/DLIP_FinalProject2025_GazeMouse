#!/usr/bin/env python
"""train_fgi_net.py — One‑file trainer for FGI‑Net on original MPIIGaze.
Run:
    python train_fgi_net.py
Override hyper‑params via env, e.g.:
    EPOCHS=60 LR=1e-4 python train_fgi_net.py
Defaults → 20‑epoch smoke‑test on a single GPU.
"""
import math, os, random, time, warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="torch"  # suppress AMP deprecation chatter
)

try:
    from timm.models.swin_transformer import SwinTransformer
except ImportError:
    raise SystemExit("[error] timm 0.9+ required → pip install timm -U")

# ── Hyper‑parameters (env‑override) ─────────────────────
ROOT        = Path(os.getenv("ROOT", Path(__file__).parent / "mpiigaze")).expanduser()
EPOCHS      = int(os.getenv("EPOCHS", 50))
BATCH       = int(os.getenv("BATCH", 256))
SPLIT       = float(os.getenv("SPLIT", 0.1))
LR          = float(os.getenv("LR", 2e-5))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 8))
USE_AMP     = os.getenv("AMP", "1") != "0"  # default ON

IMAGE_EXTS = (".jpg", ".png", ".jpeg", ".JPG")

# ── Helper functions ───────────────────────────────────

def find_image(day: Path, idx: int) -> Path:
    """Find image whose numeric stem equals idx or idx+1."""
    cand  = {idx, idx + 1}
    stems = [f"{n:0{d}d}" for n in cand for d in (3, 4, 5)]
    stems += [f"frame_{n:04d}" for n in cand] + [f"image_{n:04d}" for n in cand]
    for ext in IMAGE_EXTS:
        for s in stems:
            p = day / f"{s}{ext}"
            if p.exists():
                return p.relative_to(ROOT)
    for f in day.iterdir():
        if f.suffix.lower() in IMAGE_EXTS and f.stem.isdigit() and int(f.stem) in cand:
            return f.relative_to(ROOT)
    raise FileNotFoundError(idx)


def build_index():
    pairs = []
    for ann in ROOT.glob("p??/day??/annotation.txt"):
        day = ann.parent
        with ann.open() as fh:
            for idx, line in enumerate(fh):
                vals = line.split()
                if len(vals) < 41:
                    continue
                rel = find_image(day, idx)
                Gx, Gy, Gz = map(float, vals[26:29])
                Rx, Ry, Rz = map(float, vals[35:38])
                dx, dy, dz = Gx - Rx, Gy - Ry, Gz - Rz
                norm = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-8
                yaw   = math.atan2(dx, dz)
                pitch = math.asin(-dy / norm)
                pairs.append((str(rel), pitch, yaw))
    random.shuffle(pairs)
    cut = int(len(pairs) * (1 - SPLIT))
    return pairs[:cut], pairs[cut:]

# ── Dataset ─────────────────────────────────────────────
class GazeDS(Dataset):
    def __init__(self, tuples, aug=False):
        base = [
            transforms.Resize((72, 120)),
            transforms.Pad((0, 24, 0, 24), fill=128),
            transforms.Grayscale(3),
            transforms.ToTensor(),
        ]
        if aug:
            self.tfm = transforms.Compose([
                transforms.RandomRotation(8, fill=128),
                transforms.ColorJitter(0.2, 0.2),
                *base,
            ])
        else:
            self.tfm = transforms.Compose(base)
        self.tuples = tuples

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, i):
        rel, p, y = self.tuples[i]
        img = Image.open(ROOT / rel).convert("RGB")
        return self.tfm(img), torch.tensor([p, y], dtype=torch.float32)

# ── Model ───────────────────────────────────────────────
class FGINet(nn.Module):
    """Swin‑Tiny backbone with built‑in global pooling → (B,768) feature."""

    def __init__(self):
        super().__init__()
        self.backbone = SwinTransformer(
            img_size=120, patch_size=4, window_size=7,
            embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
            num_classes=0, global_pool="avg",
        )
        self.head = nn.Sequential(nn.Linear(768, 128), nn.ReLU(True), nn.Linear(128, 2))

    def forward(self, x):
        return self.head(self.backbone(x))


def ang_err(pred: torch.Tensor, gt: torch.Tensor):
    """Angular error (deg) between predicted (pitch,yaw) and ground truth."""
    v1 = torch.stack([
        torch.cos(gt[:, 0]) * torch.sin(gt[:, 1]),
        torch.sin(gt[:, 0]),
        torch.cos(gt[:, 0]) * torch.cos(gt[:, 1]),
    ], 1)
    v2 = torch.stack([
        torch.cos(pred[:, 0]) * torch.sin(pred[:, 1]),
        torch.sin(pred[:, 0]),
        torch.cos(pred[:, 0]) * torch.cos(pred[:, 1]),
    ], 1)
    return torch.rad2deg(torch.acos((v1 * v2).sum(1).clamp(-1, 1)))

# ── Train / Val pass ───────────────────────────────────

def run(loader, model, opt=None, scaler=None, dev="cuda"):
    """One epoch pass. Shows tqdm progress bar with live loss."""
    model.train() if opt else model.eval()
    phase    = "train" if opt else "val "
    iterator = tqdm(loader, leave=False, ncols=80, desc=phase)

    tot_l = tot_a = n = 0
    for img, tgt in iterator:
        img, tgt = img.to(dev), tgt.to(dev)
        if opt:
            opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            out  = model(img)
            loss = F.huber_loss(out, tgt) + 0.1 * F.mse_loss(out, tgt)
        if opt:
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward(); opt.step()
        tot_l += loss.item() * img.size(0)
        tot_a += ang_err(out.detach(), tgt).sum().item(); n += img.size(0)
        iterator.set_postfix(loss=f"{loss.item():.3f}")

    iterator.close()
    return tot_l / n, tot_a / n

# ── Main ────────────────────────────────────────────────

def main():
    print(f"[CFG] root={ROOT}  epochs={EPOCHS}  batch={BATCH}  split={SPLIT}")
    if not ROOT.exists():
        raise SystemExit("[error] mpiigaze folder not found; set ROOT env or place it next to script.")

    train_data, val_data = build_index()
    print(f"[DATA] total {len(train_data)+len(val_data)} • train {len(train_data)} • val {len(val_data)}")

    tr_ld = DataLoader(GazeDS(train_data, True),  BATCH, True,  num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=(NUM_WORKERS>0))
    va_ld = DataLoader(GazeDS(val_data, False), BATCH, False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=(NUM_WORKERS>0))

    dev   = "cuda" if torch.cuda.is_available() else "cpu"
    model = FGINet().to(dev)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best = float("inf")
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_l, tr_a = run(tr_ld, model, opt, scaler, dev)
        va_l, va_a = run(va_ld, model, None, None, dev)
        print(f"Ep{ep:03d}/{EPOCHS} {time.time()-t0:.1f}s  train {tr_a:.2f}°  val {va_a:.2f}°")
        
        if va_a < best:
            best = va_a
            torch.save({'state': model.state_dict(), 'best_deg': best, 'ep': ep}, 'fgi_best.pt')
            print(f"   ↳ new best saved ({best:.2f}°)")

if __name__ == "__main__":
    main()
