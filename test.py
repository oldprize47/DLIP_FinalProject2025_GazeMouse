# ──────────────── 기본 라이브러리 ────────────────
import os, time
import numpy as np
from math import hypot
from PIL import Image

# ──────────────── 컴퓨터 비전 / 머신러닝 ────────────────
import cv2                                    # OpenCV ― 영상 캡처·처리
import mediapipe as mp                        # 얼굴/눈 랜드마크
import torch
from torchvision import transforms

# ──────────────── GUI·인풋 제어 ────────────────
import pyautogui                              # 마우스 이동·클릭
pyautogui.FAILSAFE = True                     # ↖ 구석으로 이동 시 즉시 종료

# ──────────────── 모델·데이터셋 로더 ────────────────
from fginet import FGINet
from eye_patch_dataset import get_infer_transform


# ========== Mediapipe 준비 ==========
mp_face = mp.solutions.face_mesh
face = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
LEFT = [33, 133, 160, 159, 158, 157, 173, 246]
RIGHT = [362, 263, 387, 386, 385, 384, 398, 466]

# ========== 환경 ==========
calib_file = "calib_SH.npy"
CKPT = "2finetuned_SH.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOOTH_ALPHA = 0.92
model = FGINet().to(DEVICE).eval()
ckpt = torch.load(CKPT, map_location=DEVICE)
if isinstance(ckpt, dict) and "model" in ckpt:
    model.load_state_dict(ckpt["model"])
else:
    model.load_state_dict(ckpt)
tf = get_infer_transform()
W, H = pyautogui.size()
print(f"W: {W}, H: {H}")
cx, cy = W // 2, H // 2

# ================= Eye blink 환경 부분 ================================

def calculate_ear(eye_points, landmarks):
    p1, p2, p3, p4, p5, p6 = [landmarks[idx] for idx in eye_points]
    ear = (hypot(p2[0]-p6[0], p2[1]-p6[1]) + hypot(p3[0]-p5[0], p3[1]-p5[1])) / (2.0 * hypot(p1[0]-p4[0], p1[1]-p4[1]))
    return ear

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
EAR_THRESHOLD = 0.20
FIX_RADIUS = 60   # 마우스 고정 원 반경(px)
FIX_TIME = 3.0    # 고정 트리거 시간(초)
UNLOCK_EYE_TIME = 1.0  # 눈 감고 있을 때 해제 시간(초) (1초로 변경)
BLINK_COOLTIME = 0.8   # 클릭 쿨타임(초)
CONSEC_FRAMES = 2      # 블링크 프레임

class FaceMeshGenerator:
    def __init__(self, mode=False, num_faces=2, min_detection_con=0.5, min_track_con=0.5):
        self.results = None
        self.mode = mode
        self.num_faces = num_faces
        self.min_detection_con = min_detection_con
        self.min_track_con = min_track_con
        self.mp_faceDetector = mp.solutions.face_mesh
        self.face_mesh = self.mp_faceDetector.FaceMesh(
            static_image_mode=self.mode,
            max_num_faces=self.num_faces,
            min_detection_confidence=self.min_detection_con,
            min_tracking_confidence=self.min_track_con
        )
        self.mp_Draw = mp.solutions.drawing_utils
        self.drawSpecs = self.mp_Draw.DrawingSpec(thickness=1, circle_radius=2)

    def create_face_mesh(self, frame, draw=True):
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(frame_rgb)
        landmarks_dict = {}
        if self.results.multi_face_landmarks:
            for face_lms in self.results.multi_face_landmarks:
                if draw:
                    self.mp_Draw.draw_landmarks(
                        frame,
                        face_lms,
                        self.mp_faceDetector.FACEMESH_CONTOURS,
                        self.drawSpecs,
                        self.drawSpecs
                    )
                ih, iw, _ = frame.shape
                for ID, lm in enumerate(face_lms.landmark):
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    landmarks_dict[ID] = (x, y)
        return frame, landmarks_dict

def is_inside_circle(center, point, r):
    return (center[0] - point[0])**2 + (center[1] - point[1])**2 <= r**2

# =================================================

def crop_eyes(frame, face_mesh=face, img_size=224, margin=0.6):
    h, w = frame.shape[:2]
    res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:               # 얼굴 미검출
        return None, {}

    lm = res.multi_face_landmarks[0].landmark

    # 눈 둘레 중심·크기 계산 (LEFT+RIGHT 는 8-point 목록)
    xs = [lm[i].x for i in LEFT + RIGHT]
    ys = [lm[i].y for i in LEFT + RIGHT]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    cx_lm, cy_lm = (xmin + xmax) / 2, (ymin + ymax) / 2
    w_box = (xmax - xmin) * (margin + 1)
    h_box = (ymax - ymin) * (margin + 4.5)         # 세로 여유를 더 줌

    xmin, xmax = cx_lm - w_box / 2, cx_lm + w_box / 2
    ymin, ymax = cy_lm - h_box / 2, cy_lm + h_box / 2
    x1, x2 = int(max(0, xmin * w)), int(min(w - 1, xmax * w))
    y1, y2 = int(max(0, ymin * h)), int(min(h - 1, ymax * h))

    eye_patch = frame[y1:y2, x1:x2]
    if eye_patch.size == 0:                         # 드물게 잘림이 틀어질 경우
        return None, {}

    # 패치 리사이즈 → 정사각형 캔버스
    ph, pw = eye_patch.shape[:2]
    scale = img_size / max(ph, pw)
    resized = cv2.resize(eye_patch, (int(pw * scale), int(ph * scale)))
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    y_off = (img_size - resized.shape[0]) // 2
    x_off = (img_size - resized.shape[1]) // 2
    canvas[y_off:y_off + resized.shape[0], x_off:x_off + resized.shape[1]] = resized

    # 랜드마크 픽셀 좌표 dict
    lm_dict = {idx: (int(pt.x * w), int(pt.y * h)) for idx, pt in enumerate(lm)}

    return canvas, lm_dict


# ========== 캘리브레이션 지점 ==========
A = None
calibrated = False

# 1) 파일 있으면 바로 로드
if os.path.exists(calib_file):
    A = np.load(calib_file)
    calibrated = True
    print("캘리브레이션 파일을 불러왔습니다!")
margin = 0.01
x_ratios = np.linspace(0 + margin, 1 - margin, 7)
y_ratios = np.linspace(0 + margin, 1 - margin, 5)
calib_targets = [(int(round(x * (W - 1))), int(round(y * (H - 1)))) for y in y_ratios for x in x_ratios]


# ========== 캘리브레이션 함수 ==========
def calibrate(cap, min_time=2, required_stable=20, std_threshold=30):
    preds, targets = [], []
    cv2.namedWindow("calib_full", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("calib_full", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print("\n===== Calibration Mode =====")
    print("각 점에 '빨간 점'이 나오면 그 점을 바라봐 주세요!")
    print("잠시(1.5초~)동안 충분히 응시하면 다음 점으로 넘어갑니다.\n")
    GRAY_COLOR = (100, 100, 100)
    BLACK_COLOR = (0, 0, 0)
    WHITE_COLOR = (255, 255, 255)
    for idx, (tx, ty) in enumerate(calib_targets):
        bg = np.full((H, W, 3), GRAY_COLOR, dtype=np.uint8)
        cv2.circle(bg, (tx, ty), 30, (0, 0, 255), -1)
        line_len = 30  # 십자선 길이(원 크기보다 약간 길게)
        cv2.line(bg, (tx - line_len, ty), (tx + line_len, ty), BLACK_COLOR, 5)  # 가로선(검정, 두께5)
        cv2.line(bg, (tx, ty - line_len), (tx, ty + line_len), BLACK_COLOR, 5)  # 세로선(검정, 두께5)
        cv2.circle(bg, (tx, ty), 4, WHITE_COLOR, -1)
        line_len = 30  # 십자선 길이(원 크기보다 약간 길게)
        buf = []
        t_start = time.time()
        last_print = time.time()
        print(f"[{idx+1}/{len(calib_targets)}] {tx},{ty} 위치를 바라보세요 (점이 초록색으로 바뀌면 성공!)")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            patch = crop_eyes(frame)
            if patch is None:
                cv2.imshow("calib_full", bg)
                cv2.waitKey(1)
                continue
            pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
            inp = tf(pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = model(inp).cpu().squeeze().numpy()
            buf.append(pred)
            if len(buf) > required_stable:
                buf = buf[-required_stable:]  # 최신 required_stable개만 유지

            now = time.time()
            # **1초마다 프린트**
            # if now - last_print > 1.0 and len(buf) > 3:
            #     arr = np.array(buf)
            #     mean = arr.mean(axis=0)
            #     std = arr.std(axis=0)
            #     print(f"  예측값 분포 (최근 {len(buf)}개):")
            #     print(f"    X = {arr[:,0].min():.1f} ~ {arr[:,0].max():.1f},  평균 {mean[0]:.1f}, std {std[0]:.2f}")
            #     print(f"    Y = {arr[:,1].min():.1f} ~ {arr[:,1].max():.1f},  평균 {mean[1]:.1f}, std {std[1]:.2f}")
            #     last_print = now

            if (now - t_start > min_time) and len(buf) == required_stable:
                arr = np.array(buf)
                std = arr.std(axis=0)
                if np.all(std < std_threshold):
                    bg2 = bg.copy()
                    cv2.circle(bg2, (tx, ty), 30, (0, 255, 0), -1)
                    cv2.imshow("calib_full", bg2)
                    cv2.waitKey(400)
                    break

            cv2.imshow("calib_full", bg)
            # cv2.imshow("patch", patch)
            if cv2.waitKey(1) & 0xFF == 27:
                print("캘리브레이션 취소!")
                exit(0)
        mean_pred = np.mean(buf, axis=0)
        preds.append(mean_pred)
        targets.append([tx - cx, ty - cy])
    cv2.destroyWindow("calib_full")
    # cv2.destroyWindow("patch")
    # 최소제곱 보정
    preds = np.array(preds)
    targets = np.array(targets)
    ones = np.ones((preds.shape[0], 1))
    X = np.hstack([preds, ones])
    A, _, _, _ = np.linalg.lstsq(X, targets, rcond=None)
    return A


def apply_calib(pred, A):
    # pred: (2,), A: (3,2)
    x = np.append(pred, 1.0)  # [dx, dy, 1]
    offset = np.dot(x, A)  # (2,) [tx-cx, ty-cy]
    screen_xy = np.array([cx, cy]) + offset
    return screen_xy


def draw_face_body_mask(img, alpha=0.7, face_radius_ratio=0.34, body_width_ratio=0.4, body_height_ratio=0.3):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # 1. 얼굴 원 (조금 작게, 중앙 약간 위로)
    face_center = (w // 2, int(h * 0.5))  # 약간 위
    face_radius = int(min(w, h) * face_radius_ratio)
    cv2.circle(mask, face_center, face_radius, 255, -1)

    # 2. 몸통 타원 (중앙 약간 아래, 얼굴 원과 연결)
    body_center = (w // 2, int(h))
    body_axes = (int(w * body_width_ratio), int(h * body_height_ratio))
    cv2.ellipse(mask, body_center, body_axes, 0, 0, 360, 255, -1)

    # 3. 합치기: 원본은 보이고 나머지 반투명
    mask_inv = cv2.bitwise_not(mask)
    colored = np.zeros_like(img)
    colored[:] = (0, 0, 0)

    overlay = cv2.addWeighted(img, 1 - alpha, colored, alpha, 0)
    result = img.copy()
    # 반투명 영역 적용
    result[mask_inv > 0] = overlay[mask_inv > 0]
    return result


# ========== 실시간 루프 ==========

def main():
    global A, calibrated
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    pyautogui.FAILSAFE = False

    prev = np.zeros(2, np.float32)  # 시선 스무딩용
    fix_center = stay_timer = None
    fixed = False
    unlock_eye_timer = None

    blink_count = frame_counter = 0
    last_blink_time = 0

    print("[i] c: 새 캘리브레이션  |  space: 저장된 값 사용  |  esc: 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        key = cv2.waitKey(1) & 0xFF

        # ───────── 단축키 ─────────
        if key == ord('c'):
            A = calibrate(cap)
            np.save(calib_file, A)
            calibrated = True
            print(">> 캘리브레이션 완료")
            time.sleep(0.4)
            continue
        elif key == ord(' '):
            if os.path.exists(calib_file):
                A = np.load(calib_file)
                calibrated = True
                print(">> 기존 캘리브레이션 로드")
            else:
                print("저장된 캘리브레이션 없음")
            time.sleep(0.3)
            continue
        elif key == 27:      # ESC
            break

        # ───────── 눈 패치 + 랜드마크 ─────────
        patch, lm_dict = crop_eyes(frame)  # crop_eyes가 (patch, lm_dict) 반환
        if patch is None or not calibrated:
            cv2.imshow("gaze", frame)
            continue

        # ───────── 시선 추정 & 스무딩 ─────────
        pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            pred = model(tf(pil).unsqueeze(0).to(DEVICE)).cpu().squeeze().numpy()
        smoothed = SMOOTH_ALPHA * prev + (1 - SMOOTH_ALPHA) * pred
        prev[:] = smoothed

        # gaze → 픽셀
        gaze_xy = apply_calib(smoothed, A)
        gx = int(np.clip(gaze_xy[0], 5, W - 5))
        gy = int(np.clip(gaze_xy[1], 5, H - 5))

        # ───────── EAR 기반 블링크·고정 ─────────
        mouse_x, mouse_y = pyautogui.position()

        inside = is_inside_circle(fix_center, (mouse_x, mouse_y), FIX_RADIUS) if fix_center else False
        if not fixed:
            if fix_center is None or not inside:
                fix_center = (mouse_x, mouse_y)
                stay_timer = time.time()
            elif time.time() - stay_timer > FIX_TIME:
                fixed = True
                pyautogui.moveTo(*fix_center, _pause=False)
                print(">> 포인터 고정!")

        if lm_dict and all(i in lm_dict for i in LEFT_EYE + RIGHT_EYE):
            l_ear = calculate_ear(LEFT_EYE, lm_dict)
            r_ear = calculate_ear(RIGHT_EYE, lm_dict)

        # ───────── 고정 해제 조건 ─────────
        if fixed:
            if l_ear < EAR_THRESHOLD and r_ear < EAR_THRESHOLD:
                if unlock_eye_timer is None:
                    unlock_eye_timer = time.time()
                elif time.time() - unlock_eye_timer > UNLOCK_EYE_TIME:
                    fixed = False
                    fix_center = None
                    unlock_eye_timer = None
                    print(">> 고정 해제")
            else:
                unlock_eye_timer = None

        # ───────── 블링크-클릭 : 고정 상태에서만 ─────────
        if fixed:                                 # ← 추가 조건
            if l_ear < EAR_THRESHOLD and r_ear < EAR_THRESHOLD:
                frame_counter += 1
            else:
                if (frame_counter >= CONSEC_FRAMES and
                    time.time() - last_blink_time > BLINK_COOLTIME):
                    pyautogui.click()
                    blink_count += 1
                    last_blink_time = time.time()
                frame_counter = 0
        else:
            frame_counter = 0          # 고정이 아니면 카운터 초기화

            # 텍스트 디버그
            tx, gap = frame.shape[1] - 240, 45
            cv2.putText(frame, f'Blinks: {blink_count}', (tx, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'L EAR: {l_ear:.2f}', (tx - 30, 30 + gap * 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f'R EAR: {r_ear:.2f}', (tx - 30, 30 + gap * 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 2)

        # ───────── 마우스 이동 ─────────
        if fixed and fix_center:
            pyautogui.moveTo(*fix_center, _pause=False)
            cv2.circle(frame, fix_center, FIX_RADIUS, (0, 0, 255), 3)
            cv2.putText(frame, "LOCKED!", (fix_center[0] - 50, fix_center[1] - FIX_RADIUS - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            pyautogui.moveTo(gx, gy, _pause=False)
            if fix_center:
                cv2.circle(frame, fix_center, FIX_RADIUS, (200, 200, 0), 1)

        # ───────── 시각화 & 출력 ─────────
        disp = draw_face_body_mask(frame, alpha=0.68)
        cv2.imshow("gaze", disp)

    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()