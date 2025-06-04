# 파일명: parse_rtgene_labels.py


def parse_label_combined(label_file_path):
    """
    label_combined.txt를 읽어서,
    각 줄에서 (index, eye_pitch, eye_yaw)를 반환.
    """
    gaze_list = []
    with open(label_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 예시 라인:
            # 1, [0.164865454117, -0.29910425524], [-0.119121529839, 0.162986524399], 1504190788.51
            parts = line.split(",", maxsplit=3)
            # parts = ["1", " [0.164865454117", " -0.29910425524]", " [-0.119121529839, 0.162986524399], 1504190788.51"]

            # 1) index 추출
            idx_str = parts[0].strip()
            try:
                idx = int(idx_str)
            except ValueError:
                continue  # 숫자로 변환되지 않으면 건너뜀

            # 2) “eye gaze” 부분 추출
            #    parts[3] 에서 대괄호 안의 eye pitch, yaw만 가져옴
            rest = parts[3].strip()  # e.g. "[-0.119121529839, 0.162986524399], 1504190788.51"
            if "]" not in rest:
                continue
            eye_str = rest.split("]")[0]  # "[-0.119121529839, 0.162986524399"
            eye_str = eye_str.strip().lstrip("[").strip()  # "-0.119121529839, 0.162986524399"

            eye_pitch_str, eye_yaw_str = [x.strip() for x in eye_str.split(",")]
            try:
                eye_pitch = float(eye_pitch_str)
                eye_yaw = float(eye_yaw_str)
            except ValueError:
                continue

            gaze_list.append((idx, (eye_pitch, eye_yaw)))

    return gaze_list
