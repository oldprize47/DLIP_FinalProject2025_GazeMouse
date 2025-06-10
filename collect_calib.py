# 파일명: collect_calib.py  (중앙 1점 시선 바이어스 수집)

import cv2
import json
import torch
import numpy as np
import mediapipe as mp
import pyautogui
import time
from torchvision import transforms
from fginet_eyes import FGINetEyes
import math

# ── 모델·FaceMesh 로드 ───────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FGINetEyes().to(device)
ckpt = torch.load("best_fgineteyes.pth", map_location=device)
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    sd = ckpt["model_state_dict"]
else:
    sd = ckpt
model.load_state_dict(sd)
model.eval()

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)


# ── 헬퍼: gaze만 추출 ────────────────────────────────
def capture_gaze(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    h, w = frame.shape[:2]
    # 눈 좌표
    lx, ly = int(lm[33].x * w), int(lm[33].y * h)
    rx, ry = int(lm[362].x * w), int(lm[362].y * h)
    eye_size = int(1.2 * math.hypot(rx - lx, ry - ly))
    # crop (간단하게 원본 프레임에서 슬라이스)
    left_eye = frame[ly - eye_size // 2 : ly + eye_size // 2, lx - eye_size // 2 : lx + eye_size // 2]
    right_eye = frame[ry - eye_size // 2 : ry + eye_size // 2, rx - eye_size // 2 : rx + eye_size // 2]
    # 전처리
    tf_face = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tf_eye = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    inp_f = tf_face(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    inp_l = tf_eye(cv2.cvtColor(left_eye, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    inp_r = tf_eye(cv2.cvtColor(right_eye, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    inp_e = torch.cat([inp_l, inp_r], dim=1)
    # 예측
    with torch.no_grad():
        _, _, gy, gp = model(inp_f, inp_e).cpu().numpy().flatten()
    return float(gy), float(gp)


# 안내 및 마커 표시
scr_w, scr_h = pyautogui.size()
center = (scr_w // 2, scr_h // 2)

# 풀스크린 마커 + 안내 텍스트
bg = np.zeros((scr_h, scr_w, 3), dtype=np.uint8)
# 초록 원 + 검정 십자가
cv2.circle(bg, center, 50, (0, 255, 0), -1)
cv2.line(bg, (center[0] - 60, center[1]), (center[0] + 60, center[1]), (0, 0, 0), 5)
cv2.line(bg, (center[0], center[1] - 60), (center[0], center[1] + 60), (0, 0, 0), 5)
# 안내 텍스트
text = "Look at the green circle"
cv2.putText(bg, text, (int(scr_w * 0.1), int(scr_h * 0.1)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4, cv2.LINE_AA)

cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("Calibration", bg)
# 3초간 대기 (이후에도 같은 창은 유지)
cv2.waitKey(3000)
# 이제 샘플링 시작, Calibration 창은 계속 켜 둔 상태로 samples 구하게 됩니다
print("캘리브레이션 시작...")

# 샘플 수집
samples = []
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
assert cap.isOpened(), "웹캠을 열 수 없습니다"
for _ in range(100):
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)  # 좌우 반전 보정 (필요시 주석처리)
    # 매 프레임마다 Calibration 창에 마커 재표시
    cv2.imshow("Calibration", bg)
    cv2.waitKey(1)
    gp = capture_gaze(frame)
    if gp:
        samples.append(gp)
cap.release()
cv2.destroyWindow("Calibration")

# 결과 저장
mean_gy, mean_gp = np.mean(samples, axis=0)
with open("calib_center.json", "w") as f:
    json.dump({"gaze_yaw0": mean_gy, "gaze_pitch0": mean_gp}, f, indent=2)
print(f"Saved center bias: gaze_yaw0={mean_gy:.4f}, gaze_pitch0={mean_gp:.4f}")
