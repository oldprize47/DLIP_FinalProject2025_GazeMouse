# 파일명: make_csv_center.py
import os, pandas as pd

# ── 실제 경로로 바꿔주세요 ───────────────────────────────
data_root = r"C:\Users\Sangheon\source\repos\DLIP\DLIP_FinalProject2025_GazeMouse\mpiigaze"

# subject 폴더마다 해상도 매핑 (예시)
RES_MAP = {
    "p00": (1280, 800),
    "p01": (1440, 900),
    "p02": (1280, 800),
    "p03": (1440, 900),
    "p04": (1280, 800),
    "p05": (1440, 900),
    "p06": (1680, 1050),
    "p07": (1440, 900),
    "p08": (1440, 900),
    "p09": (1440, 900),
    "p10": (1440, 900),
    "p11": (1280, 800),
    "p12": (1280, 800),
    "p13": (1280, 800),
    "p14": (1440, 900),
}

rows = []
for subj in sorted(os.listdir(data_root)):
    subj_dir = os.path.join(data_root, subj)
    if not os.path.isdir(subj_dir) or subj not in RES_MAP:
        continue

    W, H = RES_MAP[subj]
    cx, cy = W / 2.0, H / 2.0

    for day in sorted(os.listdir(subj_dir)):
        day_dir = os.path.join(subj_dir, day)
        ann_path = os.path.join(day_dir, "annotation.txt")
        if not os.path.isfile(ann_path):
            continue

        imgs = sorted(f for f in os.listdir(day_dir) if f.lower().endswith((".jpg", ".png")))

        with open(ann_path, "r") as f:
            for idx, line in enumerate(f):
                if idx >= len(imgs):
                    break
                vals = line.strip().split()
                sx = float(vals[24])
                sy = float(vals[25])
                dx = sx - cx  # 중앙 기준 x 오프셋
                dy = sy - cy  # 중앙 기준 y 오프셋

                rows.append(
                    {
                        "subject": subj,
                        "day": day,
                        "image_path": os.path.join(day_dir, imgs[idx]),
                        "screen_x": sx,
                        "screen_y": sy,
                        "width": W,
                        "height": H,
                        "dx": dx,
                        "dy": dy,
                    }
                )

df = pd.DataFrame(rows)
out_csv = os.path.join(data_root, "mpiigaze_labels_center_px.csv")
df.to_csv(out_csv, index=False)
print(f"Saved {len(df)} rows → {out_csv}")
