import os
import pandas as pd

# ── 여기를 본인 경로로 바꿔주세요 ──
data_root = r"C:\Users\Sangheon\source\repos\DLIP\DLIP_FinalProject2025_GazeMouse\mpiigaze"

rows = []
for subj in sorted(os.listdir(data_root)):
    subj_dir = os.path.join(data_root, subj)
    if not os.path.isdir(subj_dir):
        continue

    for day in sorted(os.listdir(subj_dir)):
        day_dir = os.path.join(subj_dir, day)
        ann_path = os.path.join(day_dir, "annotation.txt")
        if not os.path.isfile(ann_path):
            continue

        # 이미지 파일(0001.jpg, 0002.jpg, ...)을 이름순으로 정렬
        imgs = sorted(f for f in os.listdir(day_dir) if f.lower().endswith((".jpg", ".png")))

        with open(ann_path, "r") as f:
            for idx, line in enumerate(f):
                if idx >= len(imgs):
                    break

                vals = line.strip().split()
                # annotation.txt에서 0-based 24,25 인덱스가 screen_x, screen_y
                screen_x = float(vals[24])
                screen_y = float(vals[25])

                # 이미지 절대 경로
                img_path = os.path.join(day_dir, imgs[idx])

                rows.append({"subject": subj, "day": day, "image_path": img_path, "screen_x": screen_x, "screen_y": screen_y})

# DataFrame 생성
df = pd.DataFrame(rows)

# 정규화 (1440×900 기준)
df["norm_x"] = df["screen_x"] / 1440.0
df["norm_y"] = df["screen_y"] / 900.0

# CSV로 저장
out_csv = os.path.join(data_root, "mpiigaze_labels.csv")
df.to_csv(out_csv, index=False)
print(f"Saved {len(df)} rows →\n  {out_csv}")
