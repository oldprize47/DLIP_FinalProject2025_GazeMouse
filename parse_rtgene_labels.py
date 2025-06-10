# 파일명: parse_rtgene_labels.py

import re


def parse_label_combined(label_file_path):
    """
    label_combined.txt를 읽어서,
    각 줄에서 (index, head_pitch, head_yaw, eye_pitch, eye_yaw)를 반환.
    """
    gaze_list = []
    with open(label_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 1) index
            try:
                idx = int(line.split(",", 1)[0])
            except:
                continue

            # 2) 대괄호 안의 두 쌍: 첫째=[head_pitch,head_yaw], 둘째=[eye_pitch,eye_yaw]
            matches = re.findall(r"\[([^\]]+)\]", line)
            if len(matches) < 2:
                continue

            head_str = matches[0]  # e.g. "0.164865454117, -0.29910425524"
            eye_str = matches[1]  # e.g. "-0.119121529839, 0.162986524399"

            try:
                head_pitch, head_yaw = [float(x) for x in head_str.split(",")]
                eye_pitch, eye_yaw = [float(x) for x in eye_str.split(",")]
            except:
                continue

            gaze_list.append((idx, head_pitch, head_yaw, eye_pitch, eye_yaw))

    return gaze_list
