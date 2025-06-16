import cv2, torch, numpy as np, pyautogui, time
import mediapipe as mp
from PIL import Image
from torchvision import transforms
from eye_patch_dataset import EyePatchDataset, get_infer_transform
from fginet import FGINet
import time
import os

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


def crop_eyes(frame, face_mesh=face, img_size=224, margin=0.6):
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
    h_box = (ymax - ymin) * (margin + 4.5)
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
    global A, calibrated  # 이 줄 추가
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    prev = np.array([0.0, 0.0], dtype=np.float32)
    print("[i] c 키를 누르면 캘리브레이션 시작")
    print("[i] space: 이전 캘리브레이션 불러와서 추적 시작")
    print("[i] esc 키로 종료")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        key = cv2.waitKey(1)
        # 1. c키: 새 캘리브레이션
        if key == ord("c"):
            A = calibrate(cap)
            np.save(calib_file, A)
            calibrated = True
            print("캘리브레이션 완료! 실시간 추적 시작")
            time.sleep(0.5)
            continue

        # 2. 스페이스바: 파일 있으면 바로 불러와서 실시간 추적
        if key == ord(" "):
            if os.path.exists(calib_file):
                A = np.load(calib_file)
                calibrated = True
                print("이전 캘리브레이션 불러옴! 실시간 추적 시작")
                time.sleep(0.3)
            else:
                print("저장된 캘리브레이션 파일이 없습니다. c키로 먼저 캘리브레이션하세요!")
            continue
        if key == 27:  # ESC
            break

        patch = crop_eyes(frame)
        if patch is None or not calibrated:
            cv2.imshow("gaze", frame)
            continue
        pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        inp = tf(pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(inp).cpu().squeeze().numpy()
        smoothed = SMOOTH_ALPHA * prev + (1 - SMOOTH_ALPHA) * pred
        prev = smoothed

        # 보정 적용
        screen_xy = apply_calib(smoothed, A)
        x_px = np.clip(int(screen_xy[0]), 5, W - 5)
        y_px = np.clip(int(screen_xy[1]), 5, H - 5)
        pyautogui.moveTo(x_px, y_px, _pause=False)

        # 디버그 표시
        # draw_x = int(frame.shape[1] // 2 + smoothed[0] * (frame.shape[1] / W))
        # draw_y = int(frame.shape[0] // 2 + smoothed[1] * (frame.shape[0] / H))
        # cv2.circle(frame, (draw_x, draw_y), 6, (0, 255, 0), -1)
        masked = draw_face_body_mask(frame, alpha=0.68)  # alpha 높을수록 진해짐
        cv2.imshow("gaze", masked)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
