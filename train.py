# File: train.py

import torch
from fginet import FGINet
from train_utils import set_seed, get_loaders, train_one_epoch, save_ckpt, load_ckpt, load_pretrained_weights

# --- Config (with detailed comments) ---
CSV_PATH = "p01.csv"  # Path to training/inference CSV
PRETRAINED_PTH = "model_weights.pth"  # Initial (pretrained) weights
CKPT_BEST = "model_weights.pth"  # Path to save the best checkpoint
BATCH_SIZE = 16  # Mini-batch size
EPOCHS = 150  # Maximum number of epochs
SPLIT = (0.9, 0.05, 0.05)  # Data split ratio: (train, val, test)
SEED = 42  # Random seed for reproducibility
SPLIT_JSON = f"{SEED}_eye_patch_splits_train.json"  # Path to save split indices

LR_HEAD = 1e-3  # Learning rate for MLP head
LR_BACKBONE = 3e-4  # Learning rate for backbone
WARM_EPOCHS = 2  # Number of warmup epochs
PATIENCE = 30  # Early stop if no improvement for this many epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
torch.set_float32_matmul_precision("high")  # Use high precision for matmul (PyTorch >= 2.0)


def main():
    set_seed(SEED)
    # Prepare DataLoaders
    train_ld, val_ld, test_ld = get_loaders(CSV_PATH, SPLIT, BATCH_SIZE, SEED, SPLIT_JSON)
    model = FGINet().to(device)

    # Split model parameters for differential learning rates
    head_params, backbone_params = [], []
    for n, p in model.named_parameters():
        (head_params if "stage" not in n else backbone_params).append(p)

    # AdamW optimizer: [0]=MLP head, [1]=Backbone
    opt = torch.optim.AdamW([{"params": head_params, "lr": LR_HEAD, "initial_lr": LR_HEAD}, {"params": backbone_params, "lr": LR_BACKBONE, "initial_lr": LR_BACKBONE}], weight_decay=1e-3, fused=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS - WARM_EPOCHS, eta_min=1e-5)

    # Load pretrained weights (if available)
    load_pretrained_weights(model, PRETRAINED_PTH, device)

    # Resume from checkpoint (if exists)
    start_epoch, best = load_ckpt(model, opt, CKPT_BEST, device, LR_HEAD, LR_BACKBONE)

    no_imp = 0  # Counter for early stopping
    for ep in range(start_epoch, EPOCHS):
        print(f"\n=== Epoch {ep+1}/{EPOCHS} ===")
        print(f" current LR {opt.param_groups[0]['lr']:.2e}")
        tr_loss = train_one_epoch(model, train_ld, opt, device, train=True)
        vl_loss = train_one_epoch(model, val_ld, opt, device, train=False)

        # Warmup for first WARM_EPOCHS, then cosine schedule
        if ep < WARM_EPOCHS:
            scale = (ep + 1) / WARM_EPOCHS
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * scale
        else:
            scheduler.step()

        print(f" train {tr_loss:.4f} | val {vl_loss:.4f}")
        # Save if best validation loss
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

    # Evaluate on test set with best model
    model.load_state_dict(torch.load(CKPT_BEST, map_location=device)["model"])
    model.eval()
    test_loss = train_one_epoch(model, test_ld, None, device, train=False)
    print(f"\n=== Test MAE (px): {test_loss:.2f}")


if __name__ == "__main__":
    main()
