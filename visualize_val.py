# 파일: visualize_val.py
import os, time, math, random, numpy as np, cv2, torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from rtgene_dataset import RTGENDataset
from fginet import FGINet


# ───────── util ─────────
def draw_arrow(img_bgr, pitch, yaw, color, length=5000, thickness=2):
    import math

    """
    RT-GENE 기준:
      • pitch(+) : 아래        → y(+)
      • yaw  (+) : 왼쪽        → x(−)
    """
    h, w = img_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    dx = -length * math.sin(math.radians(yaw))  # x (좌/우)
    dy = -length * math.sin(math.radians(pitch))  # y (위/아래)

    cv2.arrowedLine(img_bgr, (cx, cy), (int(cx + dx), int(cy + dy)), color=color, thickness=thickness, tipLength=0.25)


# ───────── main ─────────
def main():
    # 1) 분할용 시드(고정)  → Train/Val 비율 동일하게 유지
    split_seed = 42

    # 2) 이번 시각화용 시드(매 실행마다 다르게)
    vis_seed = int(time.time())  # ↙︎ 원하는 방식으로 바꿔도 OK
    random.seed(vis_seed)
    np.random.seed(vis_seed)
    torch.manual_seed(vis_seed)
    torch.cuda.manual_seed_all(vis_seed)

    # ───── 데이터셋 로드 & 고정 분할 ─────
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    full_ds = RTGENDataset("RT_GENE/rtgene_original_pairs.csv", transform=tf)

    val_ratio = 0.1
    val_size = int(len(full_ds) * val_ratio)
    train_size = len(full_ds) - val_size
    _, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(split_seed))

    # ───── 이번에 볼 4장을 “밑샘플”로 선정 ─────
    rand_idx = random.sample(range(len(val_ds)), 4)  # ★ val_ds 범위에서만!
    vis_ds = Subset(val_ds, rand_idx)
    vis_loader = DataLoader(vis_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    # ───── 모델 로드 ─────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FGINet().to(device)
    ckpt = torch.load("best_fginet.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    imgs, labels = next(iter(vis_loader))
    with torch.no_grad():
        preds = model(imgs.to(device)).cpu().numpy()
    imgs = imgs.cpu()
    labels = labels.numpy()

    unnorm = lambda t: (t * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    axes = axes.flatten()
    for i in range(4):
        ax = axes[i]
        img = unnorm(imgs[i]).clamp(0, 1).permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)[:, :, ::-1].copy()  # BGR contiguous

        hy, hp, gy, gp = preds[i]
        hy_gt, hp_gt, gy_gt, gp_gt = labels[i]

        draw_arrow(img, hp, hy, (0, 255, 0))  # 예측 머리
        draw_arrow(img, gp, gy, (0, 128, 0))  # 예측 시선
        draw_arrow(img, hp_gt, hy_gt, (0, 0, 255))  # GT 머리
        draw_arrow(img, gp_gt, gy_gt, (0, 0, 128))  # GT 시선

        ax.imshow(img[:, :, ::-1])
        ax.axis("off")
        mae_h = abs(hp - hp_gt) + abs(hy - hy_gt)
        mae_g = abs(gp - gp_gt) + abs(gy - gy_gt)
        ax.set_title(f"H {mae_h:.1f}° | G {mae_g:.1f}°", fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
