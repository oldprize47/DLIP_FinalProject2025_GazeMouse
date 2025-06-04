# 파일명: make_rtgene_original_pairs_all.py

import os
import csv
from parse_rtgene_labels import parse_label_combined

if __name__ == "__main__":
    # ───────────────────────────────────────────────────
    # RT-GENE 데이터셋 최상위 폴더 (본인 환경에 맞게 수정)
    RTG_ROOT = "RT_GENE"
    # 최종 저장할 CSV 경로
    out_csv = os.path.join(RTG_ROOT, "rtgene_original_pairs.csv")
    # ───────────────────────────────────────────────────

    # 1) CSV에 쓸 행 목록 준비 (헤더 포함)
    rows = [["img_path", "eye_pitch", "eye_yaw"]]
    total_count = 0

    # 2) RTG_ROOT 안의 모든 피사체 폴더를 순회
    #    예: s000_glasses, s000_noglasses, s001_glasses, ..., s016_noglasses
    for subj in sorted(os.listdir(RTG_ROOT)):
        # “_glasses”가 이름에 없는 폴더는 무시
        if "_glasses" not in subj:
            continue

        subj_path = os.path.join(RTG_ROOT, subj)
        if not os.path.isdir(subj_path):
            continue

        inner = os.path.join(subj_path, subj)
        if not os.path.isdir(inner):
            continue

        label_file = os.path.join(inner, "label_combined.txt")
        img_folder = os.path.join(inner, "original", "face")

        if not os.path.isfile(label_file) or not os.path.isdir(img_folder):
            print(f"[Skip] {inner}에 label_combined.txt 또는 original/face 폴더가 없습니다.")
            continue

        # 3) label_combined.txt 경로
        label_file = os.path.join(inner, "label_combined.txt")
        # 4) original/face 이미지 폴더
        img_folder = os.path.join(inner, "original", "face")

        if not os.path.isfile(label_file) or not os.path.isdir(img_folder):
            print(f"[Skip] {inner}에 label_combined.txt ⦿ 원본/face 폴더가 둘 다 있어야 합니다.")
            continue

        # 5) 라벨 파싱 (parse_label_combined() 은 (idx, (pitch, yaw)) 튜플 리스트를 반환한다고 가정)
        gaze_records = parse_label_combined(label_file)
        print(f"Parsed {len(gaze_records)} entries from {label_file}")

        # 6) 파싱된 각 라벨에 대해 이미지 파일이 있는지 확인하고 행 추가
        for idx, (eye_pitch, eye_yaw) in gaze_records:
            # 가능한 파일명: face_{idx:06d}_rgb.png 또는 .jpg
            fname_png = f"face_{idx:06d}_rgb.png"
            fname_jpg = f"face_{idx:06d}_rgb.jpg"

            path_png = os.path.join(img_folder, fname_png)
            path_jpg = os.path.join(img_folder, fname_jpg)

            if os.path.exists(path_png):
                img_path = path_png
            elif os.path.exists(path_jpg):
                img_path = path_jpg
            else:
                # 이미지가 없으면 스킵
                # print(f"  [Warning] {img_folder}에 {fname_png}(.jpg) 파일이 없습니다. Skipped.")
                continue

            # Windows 경로 구분자 '\' → '/'로 바꿔 주는 것이 안전
            normalized_path = img_path.replace("\\", "/")
            rows.append([normalized_path, eye_pitch, eye_yaw])
            total_count += 1

    # 7) CSV로 저장
    print(f"총 {total_count}개의 레코드를 CSV에 저장합니다 → {out_csv}")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print("CSV complete.")
