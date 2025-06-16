# 파일: gaze_live.py
import cv2, torch, numpy as np, pyautogui
import mediapipe as mp
from PIL import Image
from torchvision import transforms
from collections import deque
from eye_patch_dataset import EyePatchDataset, EyePatchDatasetInference, get_infer_transform
from fginet import FGINet
import time

frame_count = 0
t0 = time.time()
mp_face = mp.solutions.face_mesh
face = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
LEFT = [33, 133, 160, 159, 158, 157, 173, 246]
RIGHT = [362, 263, 387, 386, 385, 384, 398, 466]


def crop_eyes(frame, face_mesh=face, img_size=224, margin=0.7):
    h, w = frame.shape[:2]
    res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None  # 탐지 실패
    lm = res.multi_face_landmarks[0].landmark

    # 두 눈 bbox
    xs = [lm[i].x for i in LEFT + RIGHT]
    ys = [lm[i].y for i in LEFT + RIGHT]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # margin 확장
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    w_box = (xmax - xmin) * (margin + 1)
    h_box = (ymax - ymin) * (margin + 6)
    # 직사각 크기는 가로 w_box, 세로 h_box
    xmin, xmax = cx - w_box / 2, cx + w_box / 2
    ymin, ymax = cy - h_box / 2, cy + h_box / 2

    # 픽셀 좌표로, 경계 클램프
    x1, x2 = int(max(0, xmin * w)), int(min(w - 1, xmax * w))
    y1, y2 = int(max(0, ymin * h)), int(min(h - 1, ymax * h))

    eye_patch = frame[y1:y2, x1:x2]

    # 가로 롱 패치 → 224×224 letter-box
    patch_h, patch_w = eye_patch.shape[:2]
    scale = img_size / max(patch_h, patch_w)
    resized = cv2.resize(eye_patch, (int(patch_w * scale), int(patch_h * scale)))
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    # 중앙 배치
    y_off = (img_size - resized.shape[0]) // 2
    x_off = (img_size - resized.shape[1]) // 2
    canvas[y_off : y_off + resized.shape[0], x_off : x_off + resized.shape[1]] = resized
    return canvas  # 224×224 BGR


# ---------- 환경 ----------
CKPT = "finetuned_SH.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOOTH_ALPHA = 0.85  # 1차 저역통과 필터 계수
# --------------------------

# 1) 모델 로드
model = FGINet().to(DEVICE).eval()
ckpt = torch.load(CKPT, map_location=DEVICE)
if isinstance(ckpt, dict) and "model" in ckpt:
    model.load_state_dict(ckpt["model"])
else:
    model.load_state_dict(ckpt)

# 2) 전처리(데이터셋 클래스의 tf 그대로 사용)
tf_infer = get_infer_transform()
# 3) 모니터 해상도
W, H = pyautogui.size()
cx, cy = W // 2, H // 2  # 중앙 픽셀
# 4) 이동 필터 초기화
prev = np.array([0.0, 0.0], dtype=np.float32)  # 화면 중앙 오프셋(0,0)

# 5) 웹캠 루프
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # 좌우 반전
    frame_count += 1

    # ------ (A) ROI 획득 : Mediapipe로 눈 패치 추출 ------
    patch = crop_eyes(frame)
    if patch is None:
        cv2.imshow("gaze", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue
    patch_pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
    inp = tf_infer(patch_pil).unsqueeze(0).to(DEVICE)  # (1,3,224,224)
    # ------ (B) 추론 ------
    with torch.no_grad():
        pred = model(inp).cpu().squeeze()  # [dx, dy] in pixel (center offset)
    pred = pred.numpy()  # 예: [-120, 60] (중앙에서 왼쪽 120, 아래로 60 px)

    # ------ (C) 저역통과 필터 ------
    smoothed = SMOOTH_ALPHA * prev + (1 - SMOOTH_ALPHA) * pred
    prev = smoothed

    # ------ (D) 화면 픽셀로 변환 & 마우스 이동 ------
    x_px = int(cx + smoothed[0])  # 중앙 기준 offset 적용
    y_px = int(cy + smoothed[1])
    pyautogui.moveTo(x_px, y_px, _pause=False)

    # 디버그 뷰 (화면 프레임 상에 마킹)
    # 프레임 해상도와 모니터 해상도가 다르면 위치가 어긋날 수 있음
    draw_x = int(frame.shape[1] // 2 + smoothed[0] * (frame.shape[1] / W))
    draw_y = int(frame.shape[0] // 2 + smoothed[1] * (frame.shape[0] / H))
    cv2.circle(frame, (draw_x, draw_y), 5, (0, 255, 0), -1)
    cv2.imshow("gaze", frame)
    # cv2.imshow("patch", patch)

    # ----- FPS 계산 ----------
    if time.time() - t0 >= 1.0:
        print("fps:", frame_count)
        frame_count = 0
        t0 = time.time()
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
