import cv2
import numpy as np
import os
from PIL import Image
import pandas as pd
import mediapipe as mp
import pyautogui
import random

# --- 저장 폴더/CSV 경로 ---
SAVE_DIR = "recalib_patches_SH"
CSV_PATH = "recalib_data_SH.csv"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 모니터 해상도 ---
W, H = pyautogui.size()
print(f"W: {W}, H: {H}")

# --- 점 개수 및 비율 자동 계산 ---
N = 400  # 목표 총 점 개수

aspect = W / H
rows = int(round((N / aspect) ** 0.5))
cols = int(round(aspect * rows))

print(f"자동 계산: cols={cols}, rows={rows}, 실제 점 개수={cols*rows}")

# --- 균일한 그리드 점 좌표 생성 (margin 두고) ---
# margin_x = int(W * 0.02)
# margin_y = int(H * 0.02)
# x_list = np.linspace(margin_x, W - margin_x, cols, dtype=int)
# y_list = np.linspace(margin_y, H - margin_y, rows, dtype=int)
# --- 균일한 그리드 점 좌표 생성 (margin X) ---
x_list = np.linspace(0, W - 1, cols, dtype=int)
y_list = np.linspace(0, H - 1, rows, dtype=int)
targets = []
for idx_col, x in enumerate(x_list):
    col_targets = [(x, y) for y in (y_list if idx_col % 2 == 0 else y_list[::-1])]
    if idx_col % 2 == 1:
        col_targets = col_targets[::-1]
    targets.extend(col_targets)
random.shuffle(targets)
print(f"총 점 개수: {len(targets)}")

# --- Mediapipe 세팅 ---
mp_face = mp.solutions.face_mesh
face = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
LEFT = [33, 133, 160, 159, 158, 157, 173, 246]
RIGHT = [362, 263, 387, 386, 385, 384, 398, 466]


def crop_eyes(frame, face_mesh=face, img_size=224, margin=0.7):
    h, w = frame.shape[:2]
    res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    xs = [lm[i].x for i in LEFT + RIGHT]
    ys = [lm[i].y for i in LEFT + RIGHT]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    w_box = (xmax - xmin) * (margin + 1)
    h_box = (ymax - ymin) * (margin + 6)
    xmin, xmax = cx - w_box / 2, cx + w_box / 2
    ymin, ymax = cy - h_box / 2, cy + h_box / 2
    x1, x2 = int(max(0, xmin * w)), int(min(w - 1, xmax * w))
    y1, y2 = int(max(0, ymin * h)), int(min(h - 1, ymax * h))
    eye_patch = frame[y1:y2, x1:x2]
    patch_h, patch_w = eye_patch.shape[:2]
    scale = img_size / max(patch_h, patch_w)
    resized = cv2.resize(eye_patch, (int(patch_w * scale), int(patch_h * scale)))
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    y_off = (img_size - resized.shape[0]) // 2
    x_off = (img_size - resized.shape[1]) // 2
    canvas[y_off : y_off + resized.shape[0], x_off : x_off + resized.shape[1]] = resized
    return canvas


def show_message_on_bg(bg, message, duration=1000):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.4
    color = (0, 0, 255)  # 빨간색
    thickness = 4
    # 텍스트 위치 계산 (중앙)
    text_size, _ = cv2.getTextSize(message, font, font_scale, thickness)
    text_x = (bg.shape[1] - text_size[0]) // 2
    text_y = (bg.shape[0] + text_size[1]) // 2
    # 배경 복사
    bg_copy = bg.copy()
    cv2.putText(bg_copy, message, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.imshow("calib_point", bg_copy)
    cv2.waitKey(duration)


# --- 패치 저장 루프 ---
if os.path.exists(CSV_PATH):
    df_old = pd.read_csv(CSV_PATH)
    patches = df_old["image_path"].tolist()
    labels = list(zip(df_old["dx"], df_old["dy"]))
    print(f"기존 CSV에서 {len(patches)}개 패치 불러옴")
else:
    patches = []
    labels = []

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

idx = 0
while idx < len(targets):
    tx, ty = targets[idx]
    bg = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.circle(bg, (tx, ty), 30, (0, 255, 0), -1)
    # ─ 중심 십자선 추가 ─
    line_len = 40
    # 수평선
    cv2.line(bg, (tx - line_len, ty), (tx + line_len, ty), (0, 0, 0), 5)
    # 수직선
    cv2.line(bg, (tx, ty - line_len), (tx, ty + line_len), (0, 0, 0), 5)

    # 전체화면 모드
    cv2.namedWindow("calib_point", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("calib_point", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    go_back = False  # <<--- z키로 뒤로 갈지 플래그
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        patch = crop_eyes(frame)
        cv2.imshow("calib_point", bg)
        if patch is not None:
            patch_show = cv2.resize(patch, (224, 224))
            cv2.imshow("patch", patch_show)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC: 전체 종료
            print("작업을 종료합니다.")
            cap.release()
            cv2.destroyAllWindows()
            # CSV 저장 후 exit
            df = pd.DataFrame({"image_path": patches, "dx": [l[0] for l in labels], "dy": [l[1] for l in labels]})
            df.to_csv(CSV_PATH, index=False)
            print("CSV 및 패치 저장 완료!")
            exit()
        elif key == ord(" "):  # 스페이스로 사진 10장 연속 저장
            saved = 0
            for i in range(10):
                ret2, frame2 = cap.read()
                if not ret2:
                    continue
                frame2 = cv2.flip(frame2, 1)
                patch2 = crop_eyes(frame2)
                if patch2 is not None:
                    patch_pil2 = Image.fromarray(cv2.cvtColor(patch2, cv2.COLOR_BGR2RGB))
                    out_path2 = os.path.join(SAVE_DIR, f"{idx:03d}_{len(patches):03d}.jpg")
                    patch_pil2.save(out_path2)
                    dx = tx - W // 2
                    dy = ty - H // 2
                    patches.append(out_path2)
                    labels.append([dx, dy])
                    saved += 1
                else:
                    print(f"  (눈 crop 실패, 저장 안 됨: {i+1}/10)")
                cv2.waitKey(100)  # 100ms씩 쉬어주기
            show_message_on_bg(bg, f"Saved {saved} images!", 100)
            break
        elif key == ord("z"):
            # 최근 데이터 삭제
            num_to_remove = min(10, len(patches))
            if num_to_remove > 0:
                for _ in range(num_to_remove):
                    del_img = patches.pop()
                    labels.pop()
                    if os.path.exists(del_img):
                        os.remove(del_img)
                if idx > 0:
                    idx -= 1  # 이전 포인트로 이동
                    show_message_on_bg(bg, "Canceled!", 1000)
                    go_back = True
                    break
            else:
                show_message_on_bg(bg, "No capture to cancel!", 1000)
    cv2.destroyWindow("patch")
    if go_back:
        continue  # <<== idx += 1 실행 안함! 즉시 바깥 while 재시작(이전 포인트)
    idx += 1

cap.release()
cv2.destroyAllWindows()

# --- CSV 저장 ---
df = pd.DataFrame({"image_path": patches, "dx": [l[0] for l in labels], "dy": [l[1] for l in labels]})
df.to_csv(CSV_PATH, index=False)
print("CSV 및 패치 저장 완료!")
