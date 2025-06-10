# 파일명: gaze_mouse.py

import cv2
import torch
import numpy as np
import math
import mediapipe as mp
import pyautogui
from torchvision import transforms
from fginet_eyes import FGINetEyes

# ── 1) 모델 & FaceMesh 로드 ─────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FGINetEyes().to(device)
# PyTorch ≥2.4: weights_only, 하위호환 처리
try:
    ckpt = torch.load("best_fgineteyes.pth", map_location=device, weights_only=True)
except TypeError:
    ckpt = torch.load("best_fgineteyes.pth", map_location=device)
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    sd = ckpt["model_state_dict"]
else:
    sd = ckpt
model.load_state_dict(sd)
model.eval()

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# ── 2) 전처리 ───────────────────────────────────────────
face_tf = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
eye_tf = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# ── 3) 헬퍼 함수 ──────────────────────────────────────
def crop_square(img, cx, cy, size):
    h, w = img.shape[:2]
    left, top = int(cx - size / 2), int(cy - size / 2)
    right, bot = int(cx + size / 2), int(cy + size / 2)
    left, top = max(0, left), max(0, top)
    right, bot = min(w, right), min(h, bot)
    return img[top:bot, left:right]


# ── 4) 스크린 정보 & 필터 ───────────────────────────────
scr_w, scr_h = pyautogui.size()


def lp_filter(prev, new, alpha=0.6):
    return alpha * prev + (1 - alpha) * new if prev is not None else new


prev_x = prev_y = None

# ── 5) 중앙 바이어스 불러오기 ───────────────────────────
bias = np.load("gaze_bias.npz")
gy0, gp0 = bias["gy0"], bias["gp0"]
# (필수) 눈 각도 범위 (rad 단위) 정의
max_yaw = 0.35  # 예: 라디안으로 ±20° 범위
max_pitch = 0.35  # 예: 라디안으로 ±20° 범위
gy0, gp0 = bias["gy0"], bias["gp0"]

# ── 6) 웹캠 루프 ────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
assert cap.isOpened(), "웹캠을 열 수 없습니다"

# 전체화면 미리보기 설정
cv2.namedWindow("Gaze Cursor Preview", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Gaze Cursor Preview", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
preview_bg = np.zeros((scr_h, scr_w, 3), dtype=np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # 좌우 반전 보정

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        lx, ly = int(lm[33].x * w), int(lm[33].y * h)
        rx, ry = int(lm[362].x * w), int(lm[362].y * h)
        eye_size = int(1.2 * math.hypot(rx - lx, ry - ly))
        pts = np.array([[int(p.x * w), int(p.y * h)] for p in lm])
        fx, fy, fw, fh = cv2.boundingRect(pts)
        cx_f, cy_f = fx + fw // 2, fy + fh // 2
        face_crop = crop_square(frame, cx_f, cy_f, max(fw, fh) * 1.1)
        left_eye = crop_square(frame, lx, ly, eye_size)
        right_eye = crop_square(frame, rx, ry, eye_size)
    else:
        face_crop = frame
        left_eye = frame
        right_eye = frame

    # 모델 입력
    inp_f = face_tf(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    inp_l = eye_tf(cv2.cvtColor(left_eye, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    inp_r = eye_tf(cv2.cvtColor(right_eye, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    inp_e = torch.cat([inp_l, inp_r], dim=1)

    # 추론
    with torch.no_grad():
        hy, hp, gy, gp = model(inp_f, inp_e).cpu().numpy().flatten()

    # 바이어스 보정 (중앙 기준)
    gy_adj = gy - gy0
    gp_adj = gp - gp0

    # trig 기반 매핑
    x = scr_w / 2 + (-gy_adj / max_yaw) * (scr_w / 2)
    y = scr_h / 2 + (gp_adj / max_pitch) * (scr_h / 2)

    # 클램프 & 필터
    x = max(0, min(scr_w - 1, x))
    y = max(0, min(scr_h - 1, y))
    cur_x = lp_filter(prev_x, x)
    cur_y = lp_filter(prev_y, y)
    prev_x, prev_y = cur_x, cur_y

    # 풀스크린 미리보기
    preview = preview_bg.copy()
    cv2.circle(preview, (int(cur_x), int(cur_y)), 30, (0, 0, 255), -1)
    cv2.putText(preview, f"({int(cur_x)},{int(cur_y)})", (int(cur_x) + 40, int(cur_y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Gaze Cursor Preview", preview)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
