# 파일명: make_rtgene_pairs_face_eye.py
import os, csv
from parse_rtgene_labels import parse_label_combined

RTG_ROOT = "RT_GENE"  # ← 본인 경로
OUT_CSV = os.path.join(RTG_ROOT, "rtgene_pairs_face_eye.csv")

rows = [["face_path", "left_eye_path", "right_eye_path", "head_pitch", "head_yaw", "gaze_pitch", "gaze_yaw"]]
total = 0

for subj in sorted(os.listdir(RTG_ROOT)):
    if "_glasses" not in subj:
        continue

    inner = os.path.join(RTG_ROOT, subj, subj)
    label_file = os.path.join(inner, "label_combined.txt")
    face_dir = os.path.join(inner, "original", "face")
    left_dir = os.path.join(inner, "original", "left")
    right_dir = os.path.join(inner, "original", "right")

    if not (os.path.isfile(label_file) and os.path.isdir(face_dir) and os.path.isdir(left_dir) and os.path.isdir(right_dir)):
        print(f"[Skip] {subj}: 필수 폴더/파일이 없습니다.")
        continue

    gaze_records = parse_label_combined(label_file)
    print(f" {subj}: {len(gaze_records)} 레코드 파싱")

    for idx, hp, hy, gp, gy in gaze_records:
        fname_face = f"face_{idx:06d}_rgb"
        fname_left = f"left_{idx:06d}_rgb"
        fname_right = f"right_{idx:06d}_rgb"

        def find_path(folder, stem):
            p = os.path.join(folder, stem + ".png")  # 확장자 고정
            return p.replace("\\", "/") if os.path.exists(p) else None

        f_path = find_path(face_dir, fname_face)
        l_path = find_path(left_dir, fname_left)
        r_path = find_path(right_dir, fname_right)

        if None in (f_path, l_path, r_path):
            # 세 컷이 모두 있을 때만 사용
            continue

        rows.append([f_path, l_path, r_path, hp, hy, gp, gy])
        total += 1

print(f"▶ 총 {total} 개 레코드 CSV 저장 → {OUT_CSV}")
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(rows)
print("CSV complete.")
