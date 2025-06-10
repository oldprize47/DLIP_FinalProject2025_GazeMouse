# infer_webcam_crop.py  (핵심 부분만 발췌)
import cv2, torch, numpy as np, math, mediapipe as mp
from torchvision import transforms
from fginet import FGINet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FGINet().to(device).eval()
model.load_state_dict(torch.load("best_fginet.pth", map_location=device)["model_state_dict"])

# ── MediaPipe 얼굴 검출 초기화 ──
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# 전처리: 이제 ToTensor + Normalize 만
tf = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def rad2deg(rad):
    DEG = 180.0 / math.pi
    deg = rad * DEG
    return deg


def crop_face(frame, box, scale=1):
    """box = [x, y, w, h] in pixel, scale>1 → 여유 여백"""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    size = max(w, h) * scale
    left = int(cx - size / 2)
    top = int(cy - size / 2)
    right = int(cx + size / 2)
    bottom = int(cy + size / 2)
    # 경계 자르기
    left, top = max(0, left), max(0, top)
    right, bottom = min(frame.shape[1], right), min(frame.shape[0], bottom)
    return frame[top:bottom, left:right]


def draw_arrow_at(img_bgr, pitch, yaw, origin, color, length=3000, thickness=2):
    """
    origin = (cx, cy) : 화살표 시작점 (얼굴 중심)
    yaw(+)  = 왼쪽, pitch(+) = 아래   [RT-GENE 기준]
    """
    cx, cy = origin
    dx = -length * math.sin(math.radians(yaw))
    dy = -length * math.sin(math.radians(pitch))
    tip = (int(cx + dx), int(cy + dy))
    cv2.arrowedLine(img_bgr, (cx, cy), tip, color, thickness, tipLength=0.25)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
assert cap.isOpened()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # --- 얼굴 검출 ---
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = detector.process(rgb)

    if res.detections:
        # 가장 확신도 높은 얼굴 하나 사용
        d = max(res.detections, key=lambda det: det.score[0])
        bbox = d.location_data.relative_bounding_box
        h, w = frame.shape[:2]
        x, y = int(bbox.xmin * w), int(bbox.ymin * h)
        bw, bh = int(bbox.width * w), int(bbox.height * h)

        face = crop_face(frame, (x, y, bw, bh), scale=0.9)
        cx, cy = x + bw // 2, y + bh // 2
    else:
        face = frame  # fallback: 전체 프레임
        H, W = frame.shape[:2]
        cx, cy = W // 2, H // 2

    # --- 전처리 & 추론 ---
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_res = cv2.resize(face_rgb, (224, 224))
    inp = tf(face_res).unsqueeze(0).to(device)

    with torch.no_grad():
        head_yaw, head_pitch, gaze_yaw, gaze_pitch = model(inp).cpu().numpy().flatten()
    head_yaw += 0.25
    head_pitch += 0.5
    gaze_yaw += 0.13
    gaze_pitch += 0.2
    # --- 디스플레이 ---
    disp = frame.copy()
    draw_arrow_at(disp, head_pitch, head_yaw, (cx, cy), (0, 255, 0))
    draw_arrow_at(disp, gaze_pitch, gaze_yaw, (cx, cy), (0, 0, 255))
    txt_H = f"H yaw:{head_yaw:+5.4f}°  pitch:{head_pitch:+5.4f}°"
    txt_G = f"G yaw:{gaze_yaw:+5.4f}°  pitch:{gaze_pitch:+5.4f}°"
    cv2.putText(disp, txt_H, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(disp, txt_G, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("Face crop + yaw/pitch", disp)

    if cv2.waitKey(1) == 27:
        break  # ESC

cap.release()
cv2.destroyAllWindows()
detector.close()
