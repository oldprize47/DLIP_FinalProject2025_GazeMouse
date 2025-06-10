# 파일명: infer_webcam_gaze.py
import cv2
import torch
import numpy as np
import math
import mediapipe as mp
from torchvision import transforms
from fginet_eyes import FGINetEyes

# ── 설정 ──────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FGINetEyes().to(device)
ckpt = torch.load("best_fgineteyes.pth", map_location=device)
# PyTorch ≥2.4 지원
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    sd = ckpt["model_state_dict"]
else:
    sd = ckpt
model.load_state_dict(sd)
model.eval()

# Mediapipe FaceMesh 초기화 (눈 랜드마크)
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# 전처리
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


def crop_square(img, cx, cy, size):
    h, w = img.shape[:2]
    left = int(cx - size / 2)
    top = int(cy - size / 2)
    right = int(cx + size / 2)
    bot = int(cy + size / 2)
    left, top = max(0, left), max(0, top)
    right, bot = min(w, right), min(h, bot)
    return img[top:bot, left:right]


# 화살표 그리기
def draw_direction(img, yaw, pitch, color, length=10000, thickness=2):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    dx = -length * math.sin(math.radians(yaw))
    dy = -length * math.sin(math.radians(pitch))
    cv2.arrowedLine(img, (cx, cy), (int(cx + dx), int(cy + dy)), color, thickness, tipLength=0.2)


# 웹캠 오픈
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "웹캠을 열 수 없습니다"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(img_rgb)

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        lx, ly = int(lm[33].x * w), int(lm[33].y * h)
        rx, ry = int(lm[362].x * w), int(lm[362].y * h)
        eye_size = int(1.2 * math.dist((lx, ly), (rx, ry)))

        pts = np.array([[int(p.x * w), int(p.y * h)] for p in lm])
        fx, fy, fw, fh = cv2.boundingRect(pts)
        cx_f, cy_f = fx + fw // 2, fy + fh // 2
        face_crop = crop_square(frame, cx_f, cy_f, max(fw, fh) * 0.8)
        left_eye = crop_square(frame, lx, ly, eye_size * 0.8)
        right_eye = crop_square(frame, rx, ry, eye_size * 0.8)
    else:
        face_crop = frame
        left_eye = frame
        right_eye = frame

    # 모델 입력 준비
    inp_face = face_tf(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    inp_leye = eye_tf(cv2.cvtColor(left_eye, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    inp_reye = eye_tf(cv2.cvtColor(right_eye, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    inp_eyes = torch.cat([inp_leye, inp_reye], dim=1)

    # 추론
    with torch.no_grad():
        pred = model(inp_face, inp_eyes).cpu().numpy().flatten()
    hy, hp, gy, gp = pred[0], pred[1], pred[2], pred[3]
    hy += 0.26
    hp -= 0.3
    gy += 0.33
    gp += 0.45

    total_y = hy + gy
    total_p = hp + gp
    # 화면 표시
    disp = frame.copy()
    # 화살표
    draw_direction(disp, hy, hp, (0, 255, 0))  # 머리
    draw_direction(disp, gy, gp, (0, 0, 255))  # 시선
    draw_direction(disp, total_y, total_p, (255, 0, 0))  # 시선
    # 텍스트 오버레이
    cv2.putText(disp, f"Head Yaw:{hy:+5.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(disp, f"Head Pitch:{hp:+5.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(disp, f"Gaze Yaw:{gy:+5.4f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(disp, f"Gaze Pitch:{gp:+5.4f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 크롭 영상도 함께 표시
    cv2.imshow("Live Gaze Demo", disp)
    cv2.imshow("Face Crop", cv2.resize(face_crop, (224, 224)))
    cv2.imshow("Left Eye", cv2.resize(left_eye, (112, 112)))
    cv2.imshow("Right Eye", cv2.resize(right_eye, (112, 112)))

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
