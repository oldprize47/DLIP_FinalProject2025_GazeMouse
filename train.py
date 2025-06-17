# train.py
import os
import torch
from fginet import FGINet
from train_utils import set_seed, get_loaders, train_one_epoch, save_ckpt, load_ckpt, load_pretrained_weights

# --- Config ---
CSV_PATH = "recalib_data_SH.csv"
PRETRAINED_PTH = "Last_MPIIGAZE.pth"  # 초기 가중치
CKPT_BEST = "3finetuned_SH.pth"  # 저장할 가중치
BATCH_SIZE = 16
EPOCHS = 150
SPLIT = (0.9, 0.05, 0.05)
SEED = 42
SPLIT_JSON = f"{SEED}_eye_patch_splits_train.json"

LR_HEAD = 1e-3
LR_BACKBONE = 3e-4
WARM_EPOCHS = 2
PATIENCE = 30  # early stop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")


def main():
    set_seed(SEED)
    train_ld, val_ld, test_ld = get_loaders(CSV_PATH, SPLIT, BATCH_SIZE, SEED, SPLIT_JSON)
    model = FGINet().to(device)
    head_params, backbone_params = [], []
    for n, p in model.named_parameters():
        (head_params if "stage" not in n else backbone_params).append(p)
    opt = torch.optim.AdamW([{"params": head_params, "lr": LR_HEAD, "initial_lr": LR_HEAD}, {"params": backbone_params, "lr": LR_BACKBONE, "initial_lr": LR_BACKBONE}], weight_decay=1e-3, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS - WARM_EPOCHS, eta_min=1e-5)

    # (1) 프리트레인 가중치 불러오기
    load_pretrained_weights(model, PRETRAINED_PTH, device)

    # (2) 이어받기
    start_epoch, best = load_ckpt(model, opt, CKPT_BEST, device, LR_HEAD, LR_BACKBONE)

    no_imp = 0
    for ep in range(start_epoch, EPOCHS):
        print(f"\n=== Epoch {ep+1}/{EPOCHS} ===")
        print(f" current LR {opt.param_groups[0]['lr']:.2e}")
        tr_loss = train_one_epoch(model, train_ld, opt, device, train=True)
        vl_loss = train_one_epoch(model, val_ld, opt, device, train=False)
        if ep < WARM_EPOCHS:
            scale = (ep + 1) / WARM_EPOCHS
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * scale
        else:
            scheduler.step()
        print(f" train {tr_loss:.4f} | val {vl_loss:.4f}")
        if vl_loss < best:
            best = vl_loss
            no_imp = 0
            save_ckpt(model, opt, ep + 1, best, CKPT_BEST)
            print("  ✓ best model saved")
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"Early-stopped (no val improve {PATIENCE} ep)")
                break
    print("Training finished. Best val loss: %.4f" % best)
    # Test set
    model.load_state_dict(torch.load(CKPT_BEST, map_location=device)["model"])
    model.eval()
    test_loss = train_one_epoch(model, test_ld, None, device, train=False)
    print(f"\n=== Test MAE (px): {test_loss:.2f}")


if __name__ == "__main__":
    main()
