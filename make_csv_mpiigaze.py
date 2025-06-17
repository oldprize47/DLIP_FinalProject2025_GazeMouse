# File: make_csv_mpiigaze.py

import os
import pandas as pd

# Path to the root directory of the mpiigaze dataset
data_root = r".\mpiigaze"

# Mapping from subject folder name to its screen resolution (width, height)
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
    # Skip if not a subject directory or not in RES_MAP
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
                sx = float(vals[24])  # Screen x-coordinate
                sy = float(vals[25])  # Screen y-coordinate
                sx = (W - 1) - sx  # Flip x (horizontal mirroring)
                dx = sx - cx  # x-offset from center
                dy = sy - cy  # y-offset from center

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

# Save as CSV
df = pd.DataFrame(rows)
out_csv = os.path.join(data_root, "mpiigaze_labels_center_px.csv")
df.to_csv(out_csv, index=False)
print(f"Saved {len(df)} rows → {out_csv}")
